"""
train TS-FidelityEnhancer
"""
import copy
from argparse import ArgumentParser

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import DatasetImporterUCR

from experiments.exp_fidelity_enhancer import ExpFidelityEnhancer
from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, save_model, get_target_ucr_dataset_names


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    return parser.parse_args()


def train_stage3(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_idx,
                 ):
    project_name = 'TimeVQVAE-stage3'

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = ExpFidelityEnhancer(dataset_name, input_length, config, n_classes)
    config_ = copy.deepcopy(config)
    config_['dataset']['dataset_name'] = dataset_name
    wandb_logger = WandbLogger(project=project_name, name=None, config=config_)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_steps=config['trainer_params']['max_steps']['stage3'],
                         devices=[gpu_device_idx,],
                         accelerator='gpu',
                         val_check_interval=config['trainer_params']['val_check_interval']['stage3'],
                         check_val_every_n_epoch=None)
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    # additional log
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    print('saving the model...')
    save_model({'fidelity_enhancer': train_exp.fidelity_enhancer}, id=dataset_name)

    # test
    print('evaluating...')
    evaluation = Evaluation(dataset_name, gpu_device_idx, config)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    _, _, x_gen = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    z_train = evaluation.compute_z('train')
    z_test = evaluation.compute_z('test')
    z_gen = evaluation.compute_z_gen(x_gen)

    # fid_train = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({'FID_train_gen': evaluation.fid_score(z_train, z_gen),
               'FID_test_gen': evaluation.fid_score(z_test, z_gen),
               'FID_train_test': evaluation.fid_score(z_train, z_test),
               'IS_mean': IS_mean,
               'IS_std': IS_std})

    # evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_visual_inspection(min(200, evaluation.X_train.shape[0]), evaluation.X_train, x_gen,
                                     'X_train vs X_gen')
    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), evaluation.X_test, x_gen, 'X_test vs X_gen')
    evaluation.log_visual_inspection(min(200, evaluation.X_train.shape[0]), evaluation.X_train, evaluation.X_test,
                                     'X_train vs X_test')

    evaluation.log_pca(min(1000, z_train.shape[0]), z_train, z_gen, ['z_train', 'z_gen'])
    evaluation.log_pca(min(1000, z_test.shape[0]), z_test, z_gen, ['z_test', 'z_gen'])
    evaluation.log_pca(min(1000, z_train.shape[0]), z_train, z_test, ['z_train', 'z_test'])

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
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage3']
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage3(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx)
        torch.cuda.empty_cache()
