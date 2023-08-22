import torch
import transformers


def load_scheduler(optimizer, num_warmup_steps: int = 0, num_training_steps: int = 0):
    """Load the scheduler for your model

    Args:
        optimizer (torch.optim.AdamW): The optimizer use for training.
        num_warmup_steps (int): A generator that we can fed the model into \
            the optimizer. Defaults to None.
        weight_decay (int): The value of weight decay. Defaults to 1e-2.

    Returns:
        torch.optim.lr_scheduler.LambdaLR : Scheduler used for training.
    """

    if num_training_steps < num_warmup_steps:
        raise ValueError("Training step should be larger than warmup step.")
    if num_training_steps < 0:
        raise ValueError("Training step should be larger than 0.")

    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
