"""
train NM (Neural Mapper)
"""
import copy
import argparse
from argparse import ArgumentParser

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom

from experiments.exp_neural_mapper import ExpNeuralMapper
from utils import get_root_dir, load_yaml_param_settings, save_model, get_target_ucr_dataset_names, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket for evaluation.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage_neural_mapper(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 feature_extractor_type:str,
                 use_custom_dataset:bool,
                 ):
    project_name = 'TimeVQVAE-stage_neural_mapper'

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpNeuralMapper(dataset_name, in_channels, input_length, config, n_classes, feature_extractor_type, use_custom_dataset)

    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_name':dataset_name, 'n_trainable_params':n_trainable_params})

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
    
    eval_device = device[0] if accelerator == 'gpu' else 'cpu'
    train_exp.search_optimal_tau(X_train=train_data_loader.dataset.X, device=eval_device, wandb_logger=wandb_logger)

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage_neural_mapper'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage_neural_mapper'],
                         check_val_every_n_epoch=None)
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    print('saving the model...')
    save_model({'neural_mapper': train_exp.neural_mapper}, id=dataset_name)

    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # config
    dataset_names = get_target_ucr_dataset_names(args)
    print(' '.join(dataset_names))
    # print('dataset_names:', dataset_names)

    # run
    for dataset_name in dataset_names:
        print('dataset_name:', dataset_name)

        # data pipeline
        batch_size = config['dataset']['batch_sizes']['stage_neural_mapper']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporterCustom(**config['dataset'])
            train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage_neural_mapper(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_ind, args.feature_extractor_type, args.use_custom_dataset)
