from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
import dotenv
import os
dotenv.load_dotenv()


model_path = 'models/qwen0.5b-QA_r8/checkpoint-256'
model_name = 'Qwen/Qwen2-0.5B-Instruct'
model_path = 'juniorrios/qwen0.5-corejur-qa-fkl-ratio0.5'
max_seq_length = 8*1024



model  = AutoModelForCausalLM.from_pretrained(
    model_name,
    #max_seq_length = max_seq_length,
    #dtype = torch.bfloat16,
    attn_implementation="flash_attention_2",
    load_in_4bit = True,
    token=os.environ['HF_KEY']
)

model = PeftModelForCausalLM.from_pretrained(model, model_path, token=os.environ['HF_KEY'])


tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_KEY'])


print(model)

print('Model Loaded')




### Data Prep
from datasets import Dataset
import pandas as pd

test_file = f"data/qa_aviacao/aviacao_23q_test.json"
df = pd.read_json(test_file)



model = torch.compile(model, mode='reduce-overhead')


def run_gen(prompt):
    inputs = tokenizer(prompt['text'], return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length).to('cuda')
    
    if len(inputs['input_ids']) > 8192:
      return {'output': ['OUTOFLIMIT']}
    
    with torch.inference_mode():
      output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    try:
      output = output[0].split('###Respostas:')[1]
      return {'output': output}
    except:
      return {'output': 'OUTOFLIMIT'}


print(f"*** RUNNING MODEL LOCATED AT: {model_path}")
print(f"** DATA PATH LOCATED AT: {test_file}")
df['len'] = df['text'].apply(lambda x: len(tokenizer(x).input_ids))
df = df[df['len'] <= max_seq_length]
dataset = Dataset.from_pandas(df)
dataset = dataset.map(run_gen, batched=False)

#save_path = f"outputs/output_datasetv{args.version}_{test_file}_{model_choosed}.json"
save_path = f"outputs/qwen0.5b-QA_r8_distiltest.json"
dataset.to_pandas().to_json(save_path)
#dataset.save_to_disk('.')
#df.to_json(save_path)




