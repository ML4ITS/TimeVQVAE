"""
Stage 1: VQ training

run `python stage1.py`
"""
from argparse import ArgumentParser
import copy

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_vq_vae import ExpVQVAE
from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import *


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    return parser.parse_args()


def train_stage1(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_idx,
                 do_validate: bool,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage1'

    # fit
    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = ExpVQVAE(input_length, config, len(train_data_loader.dataset))
    config_ = copy.deepcopy(config)
    config_['dataset']['dataset_name'] = dataset_name
    wandb_logger = WandbLogger(project=project_name, name=None, config=config_)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['stage1'],
                         devices=[gpu_device_idx,],
                         accelerator='gpu',
                         check_val_every_n_epoch=round(config['trainer_params']['max_epochs']['stage1'] / 10),
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader if do_validate else None
                )

    # additional log
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    save_model({'encoder_l': train_exp.encoder_l,
                'decoder_l': train_exp.decoder_l,
                'vq_model_l': train_exp.vq_model_l,
                'encoder_h': train_exp.encoder_h,
                'decoder_h': train_exp.decoder_h,
                'vq_model_h': train_exp.vq_model_h,
                }, id=dataset_name)


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    for dataset_name in args.dataset_names:
        # data pipeline
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage1']
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage1(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx, do_validate=True)
