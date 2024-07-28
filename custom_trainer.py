from trl import SFTTrainer, SFTConfig
from peft import PeftModelForCausalLM
import torch
from losses import skewed_forward_kl, skewed_reverse_kl, js_distance, tv_distance, reverse_kl, forward_kl
from torch import nn
from torch.utils.data import DataLoader
import wandb
import os

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
    
        model, inputs = move_to_device(model, inputs, int_device=cuda_device_student)

        labels = inputs.pop("labels")

        
        outputs = model(**inputs)
        
        logits = outputs.logits
        
        #print('CALCUALNDO A LOSS CONVENCIONAL')
        # Conventional Language modeling Loss
        #lm_loss = self.loss_func(logits.float().view(-1, logits.shape[-1]), labels.view(-1))     
        lm_loss = 0
        #print(lm_loss)   
        
        # Return input['labels'] to calculate distil loss
        inputs['label'] = labels
        
        #print('CALCUALNDO A LOSS DE DISTIL')
        
        distil_loss = self.get_distil_loss(self.distil_args, inputs, logits)
        #print('Distil loss     ', distil_loss)   
        loss = (1 - self.distil_args.kd_ratio) * lm_loss + self.distil_args.kd_ratio * distil_loss
        
                        
        wandb.log({'train/loss_lm': lm_loss, 'train/loss': loss, 'train/loss_distil': distil_loss})
        
        return (loss, outputs) if return_outputs else loss  
    
  def get_distil_loss(self, args, inputs, logits):
    
    cuda_device = 0
    cuda_device_student = 0
    
    #teacher_model, inputs = move_to_device(teacher_model, inputs, int_device=0)
         
    labels = inputs.pop('label') 
    
        
    with torch.no_grad():
        self.teacher_model.eval()
        teacher_outputs = self.teacher_model(**inputs, use_cache=False)
        teacher_logits = teacher_outputs.logits
    
    del inputs['input_ids']
    del inputs['attention_mask']
    
    torch.cuda.empty_cache()
    labels = labels.cpu()
    inputs['label'] = labels
    
    
    teacher_logits = teacher_logits.cpu()  
    logits = logits.cpu()
        
    
    
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
    torch.cuda.empty_cache()
    
    return distil_loss
    
  def train(self, resume_from_checkpoint=False):
      
        wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
        wandb.init(project=os.environ['WANDB_PROJECT']) 
        
        num_epochs = self.args.num_train_epochs
        data_loader = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, collate_fn=self.data_collator)
        data_loader = iter(data_loader)
        max_steps = int(len(self.train_dataset) * num_epochs / self.args.per_device_train_batch_size)
        if not self.optimizer:
            self.create_optimizer_and_scheduler(max_steps)
                
        for step in range(max_steps):
            try:
                batch = next(data_loader)
            except:
                # reinitialize data loader 
                data_loader = iter(data_loader)
                batch = next(data_loader)
                
                
            # Modelo 1
            self.optimizer.zero_grad()
            loss = self.compute_loss(self.model, batch)
            loss.backward()
            if (step+1) % self.args.gradient_accumulation_steps == 0:
                #print('[ INFO ] STEP')
                self.optimizer.step()
                self.lr_scheduler.step()
                
                        
            if (step+1) % int(self.args.save_steps) == 0:
                print('[ INFO ] SAVING')
                self.model.save_pretrained(self.args.output_dir + f'/checkpoint-{step+1}')
            

            print(f'Step {step+1}, Loss : {loss.item()}')
            
        self.model.save_pretrained(self.args.output_dir)
    

if __name__ == '__main__':
  
  args=SFTConfig(output_dir='copa')

  
  copa = SFTDistilTrainer(teacher_model='vapo', args=args,     formatting_func=lambda x: x,
)
  print('a')