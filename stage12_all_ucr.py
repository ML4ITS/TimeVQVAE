"""
run `python train.py`
"""
from argparse import ArgumentParser

from preprocessing.preprocess_ucr import DatasetImporterUCR
from stage1 import train_stage1
from stage2 import train_stage2
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings, get_root_dir
from run_CAS import get_target_ucr_dataset_names


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.", default=get_root_dir().joinpath('configs', 'config.yaml'))

    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--gpu_device_idx', type=int, help="GPU device index", default=0)

    parser.add_argument('--n_dataset_partitions', type=int, help='used to partition all the subset datasets into N groups so that each group can be run separately.', default=1)
    parser.add_argument('--partition_idx', type=int, help='selects one partitions among the N partitions; {0, 1, ..., n_dataset_partitions-1}', default=0)
    parser.add_argument('--data_summary_ucr_condition_query', type=str, help="query to condition `data_summary_ucr_condition_query` to narrow target subset datasets.", default='')
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # run
    dataset_names = get_target_ucr_dataset_names(args)
    print('dataset_names:', dataset_names)
    for dataset_name in dataset_names:
        # set `dataset_name`
        args.data_name = f'{dataset_name}'
        config['dataset']['dataset_name'] = dataset_name
        print('# dataset_name:', config['dataset']['dataset_name'])

        # data pipeline
        dataset_importer = DatasetImporterUCR(**config['dataset'])
        batch_size = config['dataset']['batch_sizes']['stage2']
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        # train
        train_stage1(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx, do_validate=True)
        train_stage2(config, dataset_name, train_data_loader, test_data_loader, args.gpu_device_idx, do_validate=False)
