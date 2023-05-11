import torch
import transformers


def load_scheduler(
  optimizer,
  num_warmup_steps: int = 0,
  num_training_steps: int = 0
):

  # if scheduler_name == 'step':
  #   return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step, gamma=gamma)

  # if scheduler_name == 'plateau':
  #   return torch.optim.lr_scheduler.ReduceLROnPlateau(
  #     optimizer=optimizer, patience=1, mode='max', factor=gamma, verbose=True
  #   )
  # if scheduler_name == 'linear':

  '''
    Load the scheduler for your model
    ------------------------------------------
    BertTokenizerFast is for bert -> PreTrainedTokenizerFast can use in general case 
  '''
  if num_training_steps < num_warmup_steps:
    raise ValueError('Training step should be larger than warmup step.')
  if num_training_steps == 0:
    raise ValueError('Training step should be larger than 0.')

  return transformers.get_linear_schedule_with_warmup(
    optimizer=optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps
  )
