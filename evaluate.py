"""
Stage2: prior learning

run `python stage2.py`
"""
from argparse import ArgumentParser
import copy

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess_ucr import DatasetImporterUCR
import pandas as pd

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
                 gpu_device_idx,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = ExpMaskGIT(dataset_name, input_length, config, len(train_data_loader.dataset), n_classes)
    config_ = copy.deepcopy(config)
    config_['dataset']['dataset_name'] = dataset_name
    wandb_logger = WandbLogger(project='TimeVQVAE-evaluation', name=None, config=config_)

    # additional log
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('evaluating...')
    evaluation = Evaluation(dataset_name, gpu_device_idx, config)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    _, _, x_gen = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    z_test = evaluation.compute_z_test()
    z_gen = evaluation.compute_z_gen(x_gen)
    fid = evaluation.fid_score(z_test, z_gen)
    IS_mean, IS_std = evaluation.inception_score(x_gen)
    wandb.log({'FID': fid, 'IS_mean': IS_mean, 'IS_std': IS_std})

    evaluation.log_visual_inspection(min(200, evaluation.X_test.shape[0]), x_gen)
    evaluation.log_pca(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)
    evaluation.log_tsne(min(1000, evaluation.X_test.shape[0]), x_gen, z_test, z_gen)

    # compute inherent difference between Z_train and Z_test
    z_train = evaluation.compute_z_train_test('train')
    fid_inherent_error = evaluation.fid_score(z_train, z_test)
    wandb.log({'FID-inherent_error': fid_inherent_error})
    evaluation.log_visual_inspection_train_test(min(200, evaluation.X_test.shape[0]))
    evaluation.log_pca_ztrain_ztest(min(1000, evaluation.X_test.shape[0]), z_train, z_test)

    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # dataset names
    if len(args.dataset_names) == 0:
        data_summary_ucr = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
        dataset_names = data_summary_ucr['Name'].tolist()
    else:
        dataset_names = args.dataset_names
    print('dataset_names:', dataset_names)

    for dataset_name in dataset_names:
        print('dataset_name:', dataset_name)

        # data pipeline
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage2']
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage2(config, dataset_name, train_data_loader, args.gpu_device_idx)

        # clean memory
        torch.cuda.empty_cache()

