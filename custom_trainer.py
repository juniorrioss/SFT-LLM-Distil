from trl import SFTTrainer#, SFTConfig
from peft import PeftModelForCausalLM
import torch
from losses import skewed_forward_kl, skewed_reverse_kl, js_distance, tv_distance, reverse_kl, forward_kl
from torch import nn
from torch.utils.data import DataLoader
import wandb
import os
import gc
torch.autograd.set_detect_anomaly(True)

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()




def move_to_device(model, data, int_device=0):
    if isinstance(model, PeftModelForCausalLM):
        data = {k:v.to(model.device) for k,v in data.items()}
    else:
        # TODO change to output_device ou devices_ids for data parallel
        data = {k:v.to(torch.device(int_device)) for k,v in data.items()}
        
    return model, data




class SFTDistilTrainer(SFTTrainer):
  def __init__(self, teacher_model, distil_args, **kwargs):
        
    # Init all args into SFTTrainer
    super().__init__(**kwargs)

    self.teacher_model = teacher_model    
    self.loss_func = nn.CrossEntropyLoss()
    self.distil_args = distil_args
    
  def compute_loss(self, model, inputs, return_outputs=False):

        cuda_device = 1
        cuda_device_student = 0
        #import pdb 
        #pdb.set_trace()
        model, inputs = move_to_device(model, inputs, int_device=cuda_device_student)

        #labels = inputs.pop("labels")

        outputs = self.model(**inputs)
        
        logits = outputs.logits
        
        # Conventional Language modeling Loss
        #lm_loss = self.loss_func(logits.float().view(-1, logits.shape[-1]), labels.view(-1))     
        lm_loss = outputs.loss
        #inputs['label'] = inputs['labels']
        #print('[ INFO ] LM LOSS:  ', lm_loss)
        #inputs['labels'] = labels
        distil_loss = self.get_distil_loss(self.distil_args, inputs, logits)
        
        loss = (1 - self.distil_args.kd_ratio) * lm_loss + self.distil_args.kd_ratio * distil_loss
        #loss = distil_loss
        #lm_loss = 0                
        
        return loss, distil_loss, lm_loss 
    
  def get_distil_loss(self, args, inputs, logits):
    
    cuda_device_teacher = 1
    cuda_device_student = 0
    
    #self.teacher_model, inputs = move_to_device(self.teacher_model, inputs, int_device=cuda_device_teacher)
         
    labels = inputs.pop('labels') 
    
        
    with torch.no_grad():
        self.teacher_model.eval()
        teacher_outputs = self.teacher_model(**inputs, use_cache=False)
        teacher_logits = teacher_outputs.logits
    
    #del inputs['input_ids']
    #del inputs['attention_mask']
    
    #flush()
    #labels = labels.cpu()
    inputs['labels'] = labels
    
    
    #teacher_logits = teacher_logits.cpu() 
    teacher_logits = teacher_logits[:, :, :logits.shape[2]] 
    #logits = logits.cpu()
    #logits = logits[:, :, :len(self.tokenizer)] 
    
    
    
    if "sfkl" in args.type:
        distil_loss = skewed_forward_kl(logits, teacher_logits, inputs, lam=args.skew_alpha)
    elif "srkl" in args.type:
        distil_loss = skewed_reverse_kl(logits, teacher_logits, inputs, lam=args.skew_alpha)
    elif "jsd" in args.type:
        distil_loss = js_distance(logits, teacher_logits, inputs)
    elif "tvd" in args.type:
        distil_loss = tv_distance(logits, teacher_logits, inputs)
    elif "fkl" in args.type or args.type == "kd":
        distil_loss = forward_kl(logits, teacher_logits, inputs)
    elif "rkl" in args.type:
        distil_loss = reverse_kl(logits, teacher_logits, inputs)
    else:
        raise NotImplementedError
    
    print('[ INFO ] DISTIL LOSS:    ', distil_loss)
    return distil_loss
    
  def train(self, resume_from_checkpoint=False):
      
        wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
        wandb.init(project=os.environ['WANDB_PROJECT']) 
        
        num_epochs = self.args.num_train_epochs
        data_loader_gen = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, collate_fn=self.data_collator)
        data_loader = iter(data_loader_gen)
        max_steps = int(len(self.train_dataset) * num_epochs / self.args.per_device_train_batch_size)
        if not self.optimizer:
            self.create_optimizer_and_scheduler(max_steps)
        distil_loss_mean = []
        lm_loss_mean = []
        total_loss_mean = []        
        for step in range(max_steps*self.args.gradient_accumulation_steps):
            try:
                batch = next(data_loader)
            except:
                # reinitialize data loader 
                data_loader = iter(data_loader_gen)
                batch = next(data_loader)
                
                
            
            
            loss, distil_loss, lm_loss = self.compute_loss(self.model, batch)
            #import pdb;pdb.set_trace()
            loss.backward()
            #print(loss)
            total_loss_mean.append(loss.item())
            lm_loss_mean.append(lm_loss.item())
            distil_loss_mean.append(distil_loss.item())

            if (step+1) % self.args.gradient_accumulation_steps == 0:
                print('[ INFO ] STEP')
                wandb.log({'train/loss_lm': sum(lm_loss_mean)/len(lm_loss_mean), 'train/loss': sum(total_loss_mean)/self.args.gradient_accumulation_steps, 'train/loss_distil': sum(total_loss_mean)/self.args.gradient_accumulation_steps})                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                total_loss_mean = []
                lm_loss_mean = []
                distil_loss_mean = []
                print(f'Step {step+1}, Loss : {loss}')
                #flush()
                
                        
            if (step+1) % int(self.args.save_steps) == 0:
                print('[ INFO ] SAVING')
                self.model.save_pretrained(self.args.output_dir + f'/checkpoint-{step+1}')
            

            
        self.model.save_pretrained(self.args.output_dir)
    

if __name__ == '__main__':
  
  args=SFTConfig(output_dir='copa')

  
  copa = SFTDistilTrainer(teacher_model='vapo', args=args,     formatting_func=lambda x: x,
)
  print('a')
