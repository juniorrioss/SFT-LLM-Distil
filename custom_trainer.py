from trl import SFTTrainer, SFTConfig

import torch
from losses import skewed_forward_kl, skewed_reverse_kl, js_distance, tv_distance, reverse_kl, forward_kl
from torch import nn

#

def get_distil_loss(args, teacher_model, inputs, logits):
    labels = inputs.pop('label') 
      
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**inputs, use_cache=False)
        teacher_logits = teacher_outputs.logits
    
    inputs['label'] = labels
    
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
    return distil_loss


class SFTDistilTrainer(SFTTrainer):
  def __init__(self, teacher_model, distil_args, **kwargs):
        
    # Init all args into SFTTrainer
    super().__init__(**kwargs)

    self.teacher_model = teacher_model    
    self.loss_func = nn.CrossEntropyLoss()
    self.distil_args = distil_args
    
  def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        
        logits = outputs.logits
        
        #print('CALCUALNDO A LOSS CONVENCIONAL')
        # Conventional Language modeling Loss
        lm_loss = self.loss_func(logits.float().view(-1, logits.shape[-1]), labels.view(-1))     
        
        #print(lm_loss)   
        
        # Return input['labels'] to calculate distil loss
        inputs['label'] = labels
        
        #print('CALCUALNDO A LOSS DE DISTIL')
        
        distil_loss = get_distil_loss(self.distil_args, self.teacher_model, inputs, logits)
        #print('Distil loss     ', distil_loss)   
        loss = (1 - self.distil_args.kd_ratio) * lm_loss + self.distil_args.kd_ratio * distil_loss
        
        #print('LOSS geral::::::::: ', loss)
        return (loss, outputs) if return_outputs else loss  
    

if __name__ == '__main__':
  
  args=SFTConfig(output_dir='copa')

  
  copa = SFTDistilTrainer(teacher_model='vapo', args=args,     formatting_func=lambda x: x,
)
  print('a')