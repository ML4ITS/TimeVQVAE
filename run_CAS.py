"""
CAS (= TSTR):
- Train on the Synthetic samples, and
- Test on the Real samples.

[1] Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020)."""
from argparse import ArgumentParser

import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, get_root_dir
from evaluation.cas import SyntheticDataset, ExpFCN


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.", default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--config_cas', type=str, help="Path to the config_cas data  file.", default=get_root_dir().joinpath('configs', 'config_cas.yaml'))

    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', type=int, help="GPU device index", default=0)
    parser.add_argument('--min_n_synthetic_train_samples', type=int, default=1000, help='It ensures a minimum number of a number of synthetic training set size to guarantee `y ∼ pθ(y|x) = pθ(x|y)p(y)/pθ(x)`.')

    parser.add_argument('--n_dataset_partitions', type=int, default=1, help='used to partition all the subset datasets into N groups so that each group can be run separately.')
    parser.add_argument('--partition_idx', default=0, type=int, help='selects one partitions among the N partitions; {0, 1, ..., n_dataset_partitions-1}')
    parser.add_argument('--data_summary_ucr_condition_query', type=str, default='', help="query to condition `data_summary_ucr_condition_query` to narrow target subset datasets.")
    return parser.parse_args()


def get_target_ucr_dataset_names(args, ):
    if args.dataset_names:
        dataset_names = args.dataset_names
    else:
        data_summary_ucr = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
        dataset_names = data_summary_ucr['Name'].tolist()
        if args.data_summary_ucr_condition_query:
            dataset_names = data_summary_ucr.query(args.data_summary_ucr_condition_query)['Name'].values  # e.g., "500 <= Train <= 1000" for `data_summary_ucr_condition_query`
        if args.n_dataset_partitions == 1:
            pass
        elif args.n_dataset_partitions >= 1:
            b = int(np.ceil(len(dataset_names) / args.n_dataset_partitions))
            dataset_names = dataset_names[args.partition_idx * b: (args.partition_idx + 1) * b]
        else:
            raise ValueError
    return dataset_names


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    config_cas = load_yaml_param_settings(args.config_cas)

    # pre-settings
    config['trainer_params']['gpus'] = [args.gpu_device_idx]
    config_cas['trainer_params']['gpus'] = [args.gpu_device_idx]

    # run
    dataset_names = get_target_ucr_dataset_names(args)
    for dataset_name in dataset_names:
        # set `dataset_name`
        config['dataset']['dataset_name'] = dataset_name
        # config_cas['dataset']['dataset_name'] = dataset_name

        # data pipeline
        dataset_importer = DatasetImporterUCR(**config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage2']
        real_train_data_loader, real_test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        train_data_loader = DataLoader(
            SyntheticDataset(real_train_data_loader,
                             args.min_n_synthetic_train_samples,
                             config,
                             config_cas['dataset']['batch_size'],
                             config_cas['trainer_params']['gpus'][0]),
            batch_size=config_cas['dataset']['batch_size'],
            num_workers=config_cas['dataset']['num_workers'],
            shuffle=True,
            drop_last=False)

        # fit
        train_exp = ExpFCN(config_cas, len(train_data_loader.dataset), len(np.unique(train_data_loader.dataset.Y_gen)))
        wandb_logger = WandbLogger(project='TimeVQVAE-TSTR', name=config['dataset']['dataset_name'], config=vars(args) | config_cas | config)
        trainer = pl.Trainer(logger=wandb_logger,
                             enable_checkpointing=False,
                             callbacks=[LearningRateMonitor(logging_interval='epoch')],
                             devices=config_cas['trainer_params']['gpus'],
                             accelerator='gpu',
                             max_epochs=config_cas['trainer_params']['max_epochs'])
        trainer.fit(train_exp,
                    train_dataloaders=train_data_loader,
                    val_dataloaders=real_test_data_loader, )

        # test
        trainer.test(train_exp, real_test_data_loader)

        # visual comp btn real and synthetic
        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        axes = axes.flatten()
        n_samples = min(dataset_importer.X_train.shape[0], 200)
        ind0 = np.random.randint(0, dataset_importer.X_train.shape[0], size=n_samples)
        ind1 = np.random.randint(0, train_data_loader.dataset.X_gen.shape[0], size=n_samples)

        X_train = dataset_importer.X_train[ind0]  # (n_samples len)
        Y_train = dataset_importer.Y_train[ind0]  # (n_samples 1)
        X_gen = train_data_loader.dataset.X_gen.squeeze()[ind1]  # (n_samples len)
        Y_gen = train_data_loader.dataset.Y_gen.squeeze()[ind1]  # (n_samples 1)

        axes[0].plot(X_train.T, alpha=0.1)
        axes[1].plot(X_gen.T, alpha=0.1)
        plt.tight_layout()
        wandb.log({'real vs synthetic': wandb.Image(plt)})

        # finish wandb
        wandb.finish()
