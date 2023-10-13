"""
Stage2: prior learning

run `python stage2.py`
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

from experiments.exp_maskgit import ExpMaskGIT
from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, save_model


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    return parser.parse_args()


def train_stage2(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_idx,
                 do_validate: bool,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage2'

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = ExpMaskGIT(dataset_name, input_length, config, len(train_data_loader.dataset), n_classes)
    config_ = copy.deepcopy(config)
    config_['dataset']['dataset_name'] = dataset_name
    wandb_logger = WandbLogger(project=project_name, name=None, config=config_)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['stage2'],
                         devices=[gpu_device_idx,],
                         accelerator='gpu')
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader if do_validate else None
                )

    # additional log
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    print('saving the model...')
    save_model({'maskgit': train_exp.maskgit}, id=dataset_name)

    # test
    print('evaluating...')
    input_length = train_data_loader.dataset.X.shape[-1]
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    evaluation = Evaluation(dataset_name, gpu_device_idx, config)
    _, _, x_gen = evaluation.sample(max(evaluation.X_test.shape[0], config['dataset']['batch_sizes']['stage2']),
                                    input_length,
                                    n_classes,
                                    'unconditional')
    z_test, z_gen = evaluation.compute_z(x_gen)
    fid, (z_test, z_gen) = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({'FID': fid, 'IS_mean': IS_mean, 'IS_std': IS_std})

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
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
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage2']
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage2(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx, do_validate=False)

