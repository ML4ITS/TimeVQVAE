"""
Stage2: prior learning

run `python stage2.py`
"""
import argparse
from argparse import ArgumentParser
from typing import Union
import random

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

from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, save_model


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    parser.add_argument('--use_fidelity_enhancer', type=str2bool, default=False, help='Enable fidelity enhancer')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket')
    return parser.parse_args()


def evaluate(config: dict,
             dataset_name: str,
             train_data_loader: DataLoader,
             gpu_device_idx,
             use_fidelity_enhancer:bool,
             feature_extractor_type:str,
             rand_seed:Union[int,None]=None,
             ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    if not isinstance(rand_seed, type(None)):
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    input_length = train_data_loader.dataset.X.shape[-1]
    
    # wandb init
    wandb.init(project='TimeVQVAE-evaluation', 
               config={**config, 'dataset_name': dataset_name, 'use_fidelity_enhancer':use_fidelity_enhancer, 'feature_extractor_type':feature_extractor_type})

    # test
    print('evaluating...')
    evaluation = Evaluation(dataset_name, input_length, n_classes, gpu_device_idx, config, 
                            use_fidelity_enhancer=use_fidelity_enhancer,
                            feature_extractor_type=feature_extractor_type).to(gpu_device_idx)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    (_, _, xhat), xhat_R = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    z_train = evaluation.z_train
    z_test = evaluation.z_test
    z_rec_train = evaluation.compute_z_rec('train')
    z_rec_test = evaluation.compute_z_rec('test')
    z_svq_train, x_prime_train = evaluation.compute_z_svq('train')
    z_svq_test, x_prime_test = evaluation.compute_z_svq('test')
    zhat = evaluation.compute_z_gen(xhat)
    zhat_R = evaluation.compute_z_gen(xhat_R)

    IS_mean, IS_std = evaluation.inception_score(xhat)
    wandb.log({'FID(x_train, xhat)': evaluation.fid_score(z_train, zhat),
               'FID(x_test, xhat)': evaluation.fid_score(z_test, zhat),
               'FID(x_train, x_test)': evaluation.fid_score(z_train, z_test),
               'IS_mean(xhat)': IS_mean,
               'IS_std(xhat)': IS_std})

    evaluation.log_visual_inspection(evaluation.X_train, xhat, 'X_train vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_test, xhat, 'X_test vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_train, evaluation.X_test, 'X_train vs X_test')
    evaluation.log_visual_inspection(x_prime_train, x_prime_test, 'X_prime_train & X_prime_test')
    
    evaluation.log_pca([z_train,], ['z_train',])
    evaluation.log_pca([z_test,], ['z_test',])
    evaluation.log_pca([zhat,], ['zhat',])
    evaluation.log_pca([z_svq_train,], ['z_svq_train',])
    evaluation.log_pca([z_svq_test,], ['z_svq_test',])

    evaluation.log_pca([z_train, zhat], ['z_train', 'zhat'])
    evaluation.log_pca([z_test, zhat], ['z_test', 'zhat'])
    evaluation.log_pca([z_train, z_test], ['z_train', 'z_test'])

    evaluation.log_pca([z_train, z_rec_train], ['z_train', 'z_rec_train'])
    evaluation.log_pca([z_test, z_rec_test], ['z_test', 'z_rec_test'])

    evaluation.log_pca([z_train, z_svq_train], ['z_train', 'z_svq_train'])
    evaluation.log_pca([z_test, z_svq_test], ['z_test', 'z_svq_test'])
    
    if use_fidelity_enhancer:
        IS_mean, IS_std = evaluation.inception_score(xhat_R)
        wandb.log({'FID(x_train, xhat_R)': evaluation.fid_score(z_train, zhat_R),
                   'FID(x_test, xhat_R)': evaluation.fid_score(z_test, zhat_R),
                   'IS_mean(xhat_R)': IS_mean,
                   'IS_std(xhat_R)': IS_std})
        evaluation.log_visual_inspection(evaluation.X_train, xhat_R, 'X_train vs Xhat_R')
        evaluation.log_visual_inspection(evaluation.X_test, xhat_R, 'X_test vs Xhat_R')
        evaluation.log_visual_inspection(xhat[[0]], xhat_R[[0]], 'xhat vs xhat_R', alpha=1., n_plot_samples=1)  # visaulize a single pair
        evaluation.log_pca([zhat_R,], ['zhat_R',])
        evaluation.log_pca([z_train, zhat_R], ['z_train', 'zhat_R'])
        evaluation.log_pca([z_test, zhat_R], ['z_test', 'zhat_R'])

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
        evaluate(config, dataset_name, train_data_loader, args.gpu_device_idx, args.use_fidelity_enhancer, args.feature_extractor_type)

        # clean memory
        torch.cuda.empty_cache()

