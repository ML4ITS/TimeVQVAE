"""
Stage2: prior learning

run `python stage2.py`
"""
import os
import copy
from argparse import ArgumentParser
import argparse

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom

from experiments.exp_stage2 import ExpStage2
from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    parser.add_argument('--feature_extractor_type', type=str, default='supervised_fcn', help='rocket | rocket for evaluation.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage2(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 feature_extractor_type:str,
                 use_custom_dataset:bool,
                 ):
    project_name = 'TimeVQVAE-stage2'

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpStage2(dataset_name, in_channels, input_length, config, n_classes, feature_extractor_type, use_custom_dataset)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, 
                               config={**config, 'dataset_name': dataset_name, 'n_trainable_params': n_trainable_params, 'feature_extractor_type':feature_extractor_type})
    
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
                         max_steps=config['trainer_params']['max_steps']['stage2'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
                         check_val_every_n_epoch=None,
                         # precision='bf16',
                         accumulate_grad_batches=1,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    print('saving the model...')
    if not os.path.isdir(get_root_dir().joinpath('saved_models')):
        os.mkdir(get_root_dir().joinpath('saved_models'))
    trainer.save_checkpoint(os.path.join(f'saved_models', f'stage2-{dataset_name}.ckpt'))

    # test
    print('evaluating...')
    eval_device = device[0] if accelerator == 'gpu' else 'cpu'
    evaluation = Evaluation(dataset_name, in_channels, input_length, n_classes, eval_device, config, 
                            use_neural_mapper=False,
                            feature_extractor_type=feature_extractor_type,
                            use_custom_dataset=use_custom_dataset).to(eval_device)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    (_, _, x_gen), _ = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    # z_train = evaluation.z_train
    z_test = evaluation.z_test
    z_gen = evaluation.compute_z_gen(x_gen)

    # fid_train = evaluation.fid_score(z_test, z_gen)
    wandb.log({'FID': evaluation.fid_score(z_test, z_gen)})
    if not use_custom_dataset:
        IS_mean, IS_std = evaluation.inception_score(x_gen)
        wandb.log({'IS_mean': IS_mean, 'IS_std': IS_std})


    # evaluation.log_visual_inspection(evaluation.X_train, x_gen, 'X_train vs X_gen')
    evaluation.log_visual_inspection(evaluation.X_test, x_gen, 'X_test vs Xhat')
    # evaluation.log_visual_inspection(evaluation.X_train, evaluation.X_test, 'X_train vs X_test')

    # evaluation.log_pca([z_train, z_gen], ['z_train', 'z_gen'])
    evaluation.log_pca([z_test, z_gen], ['Z_test', 'Zhat'])
    # evaluation.log_pca([z_train, z_test], ['z_train', 'z_test'])

    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # config
    dataset_names = args.dataset_names

    # run
    for dataset_name in dataset_names:
        # data pipeline
        batch_size = config['dataset']['batch_sizes']['stage2']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporterCustom(**config['dataset'])
            train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage2(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_ind, args.feature_extractor_type, args.use_custom_dataset)
