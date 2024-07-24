from unsloth import FastLanguageModel, FastLlamaModel
import torch
import argparse
from tqdm import tqdm
import os
from datasets import Dataset, load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
#from unsloth import is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModelForCausalLM
import os
from dotenv import load_dotenv
import wandb

load_dotenv()

wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
wandb.init(project=os.environ['WANDB_PROJECT'])

model_choosed = "Qwen/Qwen2-0.5B-Instruct"
model_hub_path = model_choosed

max_seq_length = 8*1024 
#max_seq_length = 12*1024
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_hub_path,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = load_in_4bit,
)

model.config.torch_dtype = torch.bfloat16

lora_config = LoraConfig(
    r = 8, # ESTAVA EM 16 Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,  # ESTAVA EM 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    #use_rslora = False,  # We support rank stabilized LoRA
    # use_dora=False,
    #init_lora_weights="pissa",
    #loftq_config = None, # And LoftQ 
)

model = PeftModelForCausalLM(model, lora_config)
model = FastLlamaModel.patch_peft_model(model)


### FOR QA

train_data_path = 'data/qa_aviacao/aviacao_23q_train.json'
df = pd.read_json(train_data_path)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
df['text'] = df['text'] + EOS_TOKEN
df['len'] = df['text'].apply(lambda x: len(tokenizer(x).input_ids))
df = df[df['len'] <= max_seq_length]
dataset = Dataset.from_pandas(df)
save_path = f"models/qwen0.5b-QA_r8"
response_template = ' ###Respostas:'



### FOR SUMMARIZATION

#dataset = load_dataset('corejur/corejur_publicacoes', token='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')['train']
# def formatting_prompts_func_summ(example):
#     #output_texts = []
#     text = example['prompt']
#     text += f"\n\n### Document: {example['document']}\n\n### Answer: {example['answer']}"
    
#     return {'text': text}

# dataset = dataset.map(formatting_prompts_func_summ, batched=False)
# dataset = dataset.map(lambda x: {'length': len(tokenizer(x['text']).input_ids)})
# dataset = dataset.filter(lambda x: x['length'] <= 8192)
# print(dataset[0]['text'])
# save_path = f"models/qwen7b-summarization_r8"
# response_template = '### Answer:'






train_args = TrainingArguments(
        per_device_train_batch_size =1,
        gradient_accumulation_steps = 8,
        warmup_ratio= 0.07,
        #max_steps = 1,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        #fp16 = not is_bfloat16_supported(),
        bf16 = True,
        logging_steps = 1,
        save_steps=100,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = save_path,
        report_to="wandb",
        #run_name='qwen7b-r16-adamw-bs16',
    )



collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = train_args,
    data_collator = collator
)

print(f"\n\n*** TRAINING MODEL LOCATED AT: {model_hub_path}***")
#print(f"*** DATA PATH TO TRAIN LOCATED AT: {train_data_path}***\n\n")
trainer.train()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("\n\n****TRAINING COMPLETE****")
print(f"MODEL SAVE PATH: {save_path}")
