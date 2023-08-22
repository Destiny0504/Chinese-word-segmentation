import torch


def load_optimizer(lr, model_param=None, weight_decay=1e-2):
    """Load optimizer

    Args:
        lr (flaot): learning rate
        model_param (generator, optional): A generator that we can fed the model into \
            the optimizer. Defaults to None.
        weight_decay (float, optional): The value of weight decay. Defaults to 1e-2.

    Returns:
        torch.optim.AdamW: optimizer used for training.
    """
    param_optimizer = list(model_param)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        lr=lr, params=optimizer_grouped_parameters, weight_decay=weight_decay
    )
