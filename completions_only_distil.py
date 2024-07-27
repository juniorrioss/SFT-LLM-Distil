from dataclasses import dataclass
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
import torch
from custom_trainer import SFTDistilTrainer
import os
import wandb
from dotenv import load_dotenv
#from unsloth import FastLanguageModel, FastLlamaModel
from peft import LoraConfig, PeftModelForCausalLM
print('[ INFO ] TORCH CUDA COUNT: ', torch.cuda.device_count())

load_dotenv()


wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
wandb.init(project=os.environ['WANDB_PROJECT'])


model_name = "Qwen/Qwen2-0.5B-Instruct"
teacher_name = "Qwen/Qwen2-7B-Instruct"


max_seq_length = 8*1024 


load_in_4bit = True

# Load models

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.float16,
                                             device_map='auto'
)


teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name,
                                             attn_implementation="flash_attention_2",
                                             torch_dtype=torch.float16,
                                             device_map='auto',
                                             load_in_4bit = load_in_4bit
)

load_in_4bit = True

# Load directly with Unsloth to download 4bit models 
# teacher_model, _ = FastLanguageModel.from_pretrained(
#     teacher_name, 
#     max_seq_length = max_seq_length,
#     dtype = torch.bfloat16,
#     load_in_4bit = load_in_4bit,    
# )

# Already trained model
#teacher_model = PeftModelForCausalLM.from_pretrained(teacher_model, 'juniorrios/qwen-7b-adapter-qa-corejur', token=os.environ['HF_KEY'])
#teacher_model = FastLlamaModel.patch_peft_model(teacher_model)

teacher_model.eval()

print('[ INFO ] Teacher model')
print(teacher_model)



# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_name,
#     max_seq_length = max_seq_length,
#     dtype = torch.bfloat16,
#     load_in_4bit = load_in_4bit,
# )

lora_config = LoraConfig(
    r = 8, # ESTAVA EM 16 Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,  # ESTAVA EM 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    #use_rslora = False,  # We support rank stabilized LoRA
    use_dora=False,
    #init_lora_weights="pissa",
    #loftq_config = None, # And LoftQ 
)

model = PeftModelForCausalLM(model, lora_config)
#model = FastLlamaModel.patch_peft_model(model)


print('[ INFO ] Student model')
print(model)




tokenizer = AutoTokenizer.from_pretrained(model_name)

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['instruction'])):
#         text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
#         output_texts.append(text)
#     return output_texts

# def formatting_prompts_func(example):
#     #output_texts = []
    
#     text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
    
#     return {'text': text}

# response_template = " ### Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)



# dataset = dataset.map(formatting_prompts_func)
# dataset = dataset.remove_columns(['instruction', 'input', 'output'])




## Datasets
### FOR QA

train_data_path = 'data/qa_aviacao/aviacao_23q_train.json'
df = pd.read_json(train_data_path)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
df['text'] = df['text'] + EOS_TOKEN
df['len'] = df['text'].apply(lambda x: len(tokenizer(x).input_ids))
df = df[df['len'] <= max_seq_length]
dataset = Dataset.from_pandas(df)
save_path = f"models/qwen0.5b-QA_r8_distil_srkl-ratio0.1"
response_template = ' ###Respostas:'




@dataclass
class DistilConfig:
    skew_alpha = 0.1
    type = 'srkl'
    kd_ratio = 0.8


distil_args = DistilConfig()

sft_config = SFTConfig(
            output_dir=save_path,
            per_device_train_batch_size=1,
            per_gpu_eval_batch_size=1, 
            dataset_text_field='text', 
            max_seq_length=max_seq_length, 
            gradient_accumulation_steps=8
    )


collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTDistilTrainer(
    teacher_model=teacher_model,
    distil_args=distil_args,
    model=model,
    train_dataset=dataset,
    args=sft_config,
    #formatting_func=formatting_prompts_func,
    data_collator=collator,
    dataset_text_field='text'
)

trainer.train()