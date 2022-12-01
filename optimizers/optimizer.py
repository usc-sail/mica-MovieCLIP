import torch
from transformers import AdamW, get_linear_schedule_with_warmup

## declaration of various optimizers ####
def optimizer_adam(model,lr,weight_decay=0):
    optim_set=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def optimizer_adamW(model,lr,weight_decay):
    optim_set=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps, # Default value
                                                num_training_steps=num_training_steps)
    return(scheduler)

def reduce_lr_on_plateau(optimizer,mode,patience):
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode=mode,patience=patience)
    return(lr_scheduler)

def steplr_scheduler(optimizer,step_size,gamma):
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return(scheduler)
