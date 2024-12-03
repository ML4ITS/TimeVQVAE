import wandb
import pytorch_lightning as pl


def detach_the_unnecessary(loss_hist: dict):
    """
    apply `.detach()` on Tensors that do not need back-prop computation.
    :return:
    """
    for k in loss_hist.keys():
        if k not in ['loss']:
            try:
                loss_hist[k] = loss_hist[k].detach()
            except AttributeError:
                pass


def compute_avg_outs(outs: dict):
    mean_outs = {}
    for k in outs[0].keys():
        mean_outs.setdefault(k, 0.)
        for i in range(len(outs)):
            mean_outs[k] += outs[i][k]
        mean_outs[k] /= len(outs)
    return mean_outs


def get_log_items_epoch(kind: str, current_epoch: int, mean_outs: dict):
    log_items_ = {f'{kind}/{k}': v for k, v in mean_outs.items()}
    log_items = {'epoch': current_epoch}
    log_items = dict(log_items.items() | log_items_.items())
    return log_items


def get_log_items_global_step(kind: str, global_step: int, out: dict):
    log_items_ = {f'{kind}/{k}': v for k, v in out.items()}
    log_items = {'global_step': global_step}
    log_items = dict(log_items.items() | log_items_.items())
    return log_items


class ExpBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def validation_step(self, batch, batch_idx):
        raise NotImplemented

    def training_epoch_end(self, outs) -> None:
        mean_outs = compute_avg_outs(outs)
        log_items = get_log_items_epoch('train', self.current_epoch, mean_outs)
        wandb.log(log_items)

    def training_step_end(self, out) -> None:
        log_items = get_log_items_global_step('train', self.global_step, out)
        wandb.log(log_items)

    def validation_epoch_end(self, outs) -> None:
        mean_outs = compute_avg_outs(outs)
        log_items = get_log_items_epoch('val', self.current_epoch, mean_outs)
        wandb.log(log_items)

    def validation_step_end(self, out) -> None:
        log_items = get_log_items_global_step('val', self.global_step, out)
        wandb.log(log_items)

    def configure_optimizers(self):
        raise NotImplemented

    def test_epoch_end(self, outs) -> None:
        mean_outs = compute_avg_outs(outs)
        log_items = get_log_items_epoch('test', self.current_epoch, mean_outs)
        wandb.log(log_items)

    def test_step_end(self, out) -> None:
        log_items = get_log_items_global_step('test', self.global_step, out)
        wandb.log(log_items)
