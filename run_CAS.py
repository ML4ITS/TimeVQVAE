"""
CAS (= TSTR):
- Train on the Synthetic samples, and
- Test on the Real samples.

[1] Smith, Kaleb E., and Anthony O. Smith. "Conditional GAN for timeseries generation." arXiv preprint arXiv:2006.16477 (2020).
"""
from argparse import ArgumentParser

import wandb
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from preprocessing.preprocess_ucr import DatasetImporterUCR
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, get_root_dir, get_target_ucr_dataset_names, str2bool
from evaluation.cas import CASDataset, ExpFCN


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.", default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--config_cas', type=str, help="Path to the config_cas data  file.", default=get_root_dir().joinpath('configs', 'config_cas.yaml'))

    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', type=int, help="GPU device index", default=0)
    parser.add_argument('--use_neural_mapper', type=str2bool, default=False, help='Use the neural mapper')
    parser.add_argument('--min_n_synthetic_train_samples', type=int, default=1000, help='It ensures a minimum number of a number of synthetic training set size to guarantee `y ∼ pθ(y|x) = pθ(x|y)p(y)/pθ(x)`.')
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    config_cas = load_yaml_param_settings(args.config_cas)
    
    device = args.gpu_device_idx
    batch_size = config_cas['dataset']['batch_size']

    # run
    dataset_names = get_target_ucr_dataset_names(args)
    print('dataset_names:', dataset_names)
    for dataset_name in dataset_names:
        # data pipeline
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
        real_train_data_loader, real_test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        synthetic_data_loader = DataLoader(
            CASDataset(real_train_data_loader,
                       args.min_n_synthetic_train_samples,
                       dataset_name,
                       config,
                       device,
                       args.use_neural_mapper),
            batch_size=batch_size,
            num_workers=config_cas['dataset']['num_workers'],
            shuffle=True,
            drop_last=False)

        # fit
        wandb_logger = WandbLogger(project='TimeVQVAE-CAS', 
                                   name=dataset_name, 
                                   config={**config, **config_cas, 'device':device, 'dataset_name':dataset_name})
        n_classes = len(np.unique(real_train_data_loader.dataset.Y))
        train_exp = ExpFCN(config_cas, n_classes)
        trainer = pl.Trainer(logger=wandb_logger,
                             enable_checkpointing=False,
                             callbacks=[LearningRateMonitor(logging_interval='step')],
                             max_steps=config_cas['trainer_params']['max_steps'],
                             devices=[device,],
                             accelerator='gpu',
                             val_check_interval=config_cas['trainer_params']['val_check_interval'],
                             check_val_every_n_epoch=None,)
        trainer.fit(train_exp,
                    train_dataloaders=synthetic_data_loader,
                    val_dataloaders=real_test_data_loader, )

        wandb.finish()
