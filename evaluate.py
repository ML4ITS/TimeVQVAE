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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_data_pipeline, build_custom_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom
import pandas as pd

from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    parser.add_argument('--use_neural_mapper', type=str2bool, default=False, help='Use the neural mapper')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket for evaluation.')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    parser.add_argument('--sampling_batch_size', type=int, default=None, help='batch size when sampling.')
    return parser.parse_args()


def evaluate(config: dict,
             dataset_name: str,
             train_data_loader: DataLoader,
             gpu_device_idx,
             use_neural_mapper:bool,
             feature_extractor_type:str,
             use_custom_dataset:bool=False,
             sampling_batch_size=None,
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
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    
    # wandb init
    wandb.init(project='TimeVQVAE-evaluation', 
               config={**config, 'dataset_name': dataset_name, 'use_neural_mapper':use_neural_mapper, 'feature_extractor_type':feature_extractor_type})

    # unconditional sampling
    print('evaluating...')
    evaluation = Evaluation(dataset_name, in_channels, input_length, n_classes, gpu_device_idx, config, 
                            use_neural_mapper=use_neural_mapper,
                            feature_extractor_type=feature_extractor_type,
                            use_custom_dataset=use_custom_dataset).to(gpu_device_idx)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    (_, _, xhat), xhat_R = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional', batch_size=sampling_batch_size)
    z_train = evaluation.z_train
    z_test = evaluation.z_test
    z_rec_train = evaluation.compute_z_rec('train')
    z_rec_test = evaluation.compute_z_rec('test')
    zhat = evaluation.compute_z_gen(xhat)
    
    # compute FID and IS
    print('evaluation for unconditional sampling...')
    wandb.log({'FID': evaluation.fid_score(z_test, zhat)})
    if not use_custom_dataset:
        IS_mean, IS_std = evaluation.inception_score(xhat)
        wandb.log({'IS_mean': IS_mean, 'IS_std': IS_std})

    evaluation.log_visual_inspection(evaluation.X_train, xhat, 'X_train vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_test, xhat, 'X_test vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_train, evaluation.X_test, 'X_train vs X_test')
    
    evaluation.log_pca([z_train,], ['Z_train',])
    evaluation.log_pca([z_test,], ['Z_test',])
    evaluation.log_pca([zhat,], ['Zhat',])

    evaluation.log_pca([z_train, zhat], ['Z_train', 'Zhat'])
    evaluation.log_pca([z_test, zhat], ['Z_test', 'Zhat'])
    evaluation.log_pca([z_train, z_test], ['Z_train', 'Z_test'])

    evaluation.log_pca([z_train, z_rec_train], ['Z_train', 'Z_rec_train'])
    evaluation.log_pca([z_test, z_rec_test], ['Z_test', 'Z_rec_test'])

    mdd, acd, sd, kd = evaluation.stat_metrics(evaluation.X_test, xhat)
    wandb.log({'MDD':mdd, 'ACD':acd, 'SD':sd, 'KD':kd})
    
    if use_neural_mapper:
        z_svq_train, x_prime_train = evaluation.compute_z_svq('train')
        z_svq_test, x_prime_test = evaluation.compute_z_svq('test')
        zhat_R = evaluation.compute_z_gen(xhat_R)
        
        evaluation.log_pca([z_svq_train,], ['Z_svq_train',])
        evaluation.log_pca([z_svq_test,], ['Z_svq_test',])
        evaluation.log_visual_inspection(x_prime_train, x_prime_test, 'X_prime_train & X_prime_test')
        evaluation.log_pca([z_train, z_svq_train], ['Z_train', 'Z_svq_train'])
        evaluation.log_pca([z_test, z_svq_test], ['Z_test', 'Z_svq_test'])

        IS_mean, IS_std = evaluation.inception_score(xhat_R)
        wandb.log({'FID with NM': evaluation.fid_score(z_test, zhat_R),
                   'IS_mean with NM': IS_mean,
                   'IS_std with NM': IS_std})
        
        evaluation.log_visual_inspection(evaluation.X_train, xhat_R, 'X_train vs Xhat_R')
        evaluation.log_visual_inspection(evaluation.X_test, xhat_R, 'X_test vs Xhat_R')
        evaluation.log_visual_inspection(xhat[[0]], xhat_R[[0]], 'xhat vs xhat_R', alpha=1., n_plot_samples=1)  # visaulize a single pair
        evaluation.log_pca([zhat_R,], ['Zhat_R',])
        evaluation.log_pca([z_train, zhat_R], ['Z_train', 'Zhat_R'])
        evaluation.log_pca([z_test, zhat_R], ['Z_test', 'Zhat_R'])

        mdd, acd, sd, kd = evaluation.stat_metrics(evaluation.X_test, xhat_R)
        wandb.log({'MDD with NM':mdd, 'ACD with NM':acd, 'SD with NM':sd, 'KD with NM':kd})
        
    # class-conditional sampling
    print('evaluation for class-conditional sampling...')
    n_plot_samples_per_class = 100 #200
    alpha = 0.1
    ylim = (-5, 5)
    n_rows = int(np.ceil(np.sqrt(n_classes)))
    fig1, axes1 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    fig2, axes2 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    fig3, axes3 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    fig1.suptitle('X_test_c')
    fig2.suptitle(f"Xhat_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
    fig3.suptitle(f"Xhat_R_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()
    axes3 = axes3.flatten()
    n_cls_samples = []
    cfids, cfids_nm = [], []
    for cls_idx in range(n_classes):
        (_, _, xhat_c), xhat_c_R = evaluation.sample(n_plot_samples_per_class, kind='conditional', class_index=cls_idx)
        
        cls_sample_ind = (evaluation.Y_test[:,0] == cls_idx)  # (b,)
        n_cls_samples.append(cls_sample_ind.astype(float).sum())

        z_test_c = evaluation.compute_z_gen(torch.from_numpy(evaluation.X_test[cls_sample_ind]))
        zhat_c = evaluation.compute_z_gen(xhat_c)
        cfid = evaluation.fid_score(z_test_c, zhat_c)
        cfids.append(cfid)
        wandb.log({f'cFID-cls_{cls_idx}': cfid})
        if use_neural_mapper:
            zhat_R = evaluation.compute_z_gen(xhat_c_R)
            cfid_nm = evaluation.fid_score(z_test_c, zhat_R)
            cfids_nm.append(cfid_nm)
            wandb.log({f'cFID+NM-cls_{cls_idx}': cfid})

        X_test_c = evaluation.X_test[cls_sample_ind]  # (b' 1 l)
        sample_ind = np.random.randint(0, X_test_c.shape[0], n_plot_samples_per_class)
        axes1[cls_idx].plot(X_test_c[sample_ind,0,:].T, alpha=alpha, color='C0')
        axes1[cls_idx].set_title(f'cls_idx:{cls_idx}')
        axes1[cls_idx].set_ylim(*ylim)

        sample_ind = np.random.randint(0, xhat_c.shape[0], n_plot_samples_per_class)
        axes2[cls_idx].plot(xhat_c[sample_ind,0,:].T, alpha=alpha, color='C0')
        axes2[cls_idx].set_title(f'cls_idx:{cls_idx}')
        axes2[cls_idx].set_ylim(*ylim)

        if use_neural_mapper:
            sample_ind = np.random.randint(0, xhat_c_R.shape[0], n_plot_samples_per_class)
            axes3[cls_idx].plot(xhat_c_R[sample_ind,0,:].T, alpha=alpha, color='C0')
            axes3[cls_idx].set_title(f'cls_idx:{cls_idx}')
            axes3[cls_idx].set_ylim(*ylim)

    fig1.tight_layout()
    fig2.tight_layout()
    wandb.log({"X_test_c": wandb.Image(fig1)})
    wandb.log({f"Xhat_c": wandb.Image(fig2)})
    wandb.log({f'cFID': np.mean(cfids)})

    if use_neural_mapper:
        fig3.tight_layout()
        wandb.log({f"Xhat_R_c": wandb.Image(fig3)})
        wandb.log({f'cFID+NM': np.mean(cfids_nm)})

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    # bar graph of cfids
    fig, ax = plt.subplots()
    ax.bar(range(n_classes), cfids)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('FID per class')
    ax.set_xticks(range(n_classes))  # Ensure x-axis labels are integers only
    wandb.log({"cFID_bar": wandb.Image(fig)})
    plt.close(fig)

    if use_neural_mapper:
        fig, ax = plt.subplots()
        ax.bar(range(n_classes), cfids_nm)
        ax.set_xlabel('Class Index')
        ax.set_ylabel('FID with NM per class')
        ax.set_xticks(range(n_classes))  # Ensure x-axis labels are integers only
        wandb.log({"cFID+FE_bar": wandb.Image(fig)})
        plt.close(fig)

    # plot bar graph of n_cls_samples
    fig, ax = plt.subplots()
    ax.bar(range(n_classes), n_cls_samples)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('num samples per class')
    ax.set_xticks(range(n_classes))  # Ensure x-axis labels are integers only
    print('n_cls_samples:', n_cls_samples)
    wandb.log({"n_cls_samples": wandb.Image(fig)})
    plt.close(fig)
    
    # compute cFID (conditional FID)
    wandb.log({'FID': evaluation.fid_score(z_test, zhat)})

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
        batch_size = config['evaluation']['batch_size']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporterCustom(**config['dataset'])
            train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        evaluate(config, dataset_name, train_data_loader, args.gpu_device_idx, args.use_neural_mapper, args.feature_extractor_type, args.use_custom_dataset, args.sampling_batch_size)

        # clean memory
        torch.cuda.empty_cache()

