import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


def linear_warmup_cosine_annealingLR(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    linear_warmup_rate: float = 0.1,
    min_lr: float = 1e-6,
):
    assert 0.0 < linear_warmup_rate < 1.0, "0 < linear_warmup_rate < 1."

    warmup_steps = int(max_steps * linear_warmup_rate)

    def warmup_lambda(current_step):
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, max_steps - warmup_steps, eta_min=min_lr)

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
