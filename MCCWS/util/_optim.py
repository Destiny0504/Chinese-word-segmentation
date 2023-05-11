import torch


def load_optimizer(lr, opt_name: str = None, model_param=None, weight_decay=1e-2):
  param_optimizer = list(model_param)
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {
      'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay': weight_decay
    }, {
      'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay': 0.0
    }
  ]
  return torch.optim.AdamW(lr=lr, params=optimizer_grouped_parameters, weight_decay=weight_decay)