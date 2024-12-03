"""
Stage 1: VQ training

run `python stage1.py`
"""
import os
from argparse import ArgumentParser
import copy

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom
from experiments.exp_stage1 import ExpStage1
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage1(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage1'

    # fit
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpStage1(in_channels, input_length, config)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_name': dataset_name, 'n_trainable_params:': n_trainable_params})

    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpu_device_ind
        
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage1'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
                         check_val_every_n_epoch=None,
                         # precision='bf16',
                         accumulate_grad_batches=1,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    if not os.path.isdir(get_root_dir().joinpath('saved_models')):
            os.mkdir(get_root_dir().joinpath('saved_models'))
    trainer.save_checkpoint(os.path.join(f'saved_models', f'stage1-{dataset_name}.ckpt'))


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    for dataset_name in args.dataset_names:
        # data pipeline
        batch_size = config['dataset']['batch_sizes']['stage1']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporterCustom(**config['dataset'])
            train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage1(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_ind)
