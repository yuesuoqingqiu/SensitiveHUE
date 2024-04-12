import os
import torch
import argparse
import sensitive_hue
import numpy as np
import torch.optim as optim
from config.parser import YAMLParser
from base.dataset import ADataset, split_dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def get_data_loaders(data_args, args, scaler):
    dataset_names = ('train.npy', 'test.npz')
    keys = (None, ('x', 'y'))

    data_loaders = []
    for name, key in zip(dataset_names, keys):
        stride = args.stride if name.startswith('train') else 1
        dataset = ADataset(os.path.join(data_args.data_dir, name), data_args.step_num_in, stride, keys=key)
        if name.startswith('train'):
            if scaler is not None:
                dataset.fit_transform(scaler.fit_transform)
            random_val = getattr(data_args, 'random_val', False)
            for d_set in split_dataset(dataset, data_args.val_ratio, random_val):
                data_loader = DataLoader(d_set, batch_size=data_args.batch_size, shuffle=True)
                data_loaders.append(data_loader)
        else:
            dataset.fit_transform(scaler.transform)
            data_loader = DataLoader(dataset, batch_size=data_args.batch_size)
            data_loaders.append(data_loader)

    return data_loaders


def train_single_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    scaler = StandardScaler()
    data_loaders = get_data_loaders(data_args, args, scaler)
    # ------------------------ Trainer Setting ----------------------------
    model = sensitive_hue.SensitiveHUE(
        data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
        data_args.dim_hidden_fc, data_args.encode_layer_num, 0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    trainer = sensitive_hue.Trainer(
        model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
        scheduler, use_prob=True)

    if not only_test:
        trainer.train(data_loaders[0], data_loaders[1])
    ignore_dims = getattr(data_args, 'ignore_dims', None)
    trainer.test(data_loaders[-1], ignore_dims, data_args.select_pos)


def train_multi_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    scaler = StandardScaler()
    start, end = data_args.range
    data_dir = data_args.data_dir

    ignore_dims = getattr(data_args, 'ignore_dims', dict())
    ignore_entities = getattr(data_args, 'ignore_entities', tuple())

    results = []
    for i in range(start, end + 1):
        if i in ignore_entities:
            continue
        data_args.data_dir = os.path.join(data_dir, f'{data_name}_{i}')
        data_loaders = get_data_loaders(data_args, args, scaler)

        model = sensitive_hue.SensitiveHUE(
            data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
            data_args.dim_hidden_fc, data_args.encode_layer_num, 0.1
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        trainer = sensitive_hue.Trainer(
            model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
            scheduler, use_prob=True, model_save_suffix=f'_{i}'
        )

        trainer.logger.info(f'entity {i}')

        if not only_test:
            trainer.train(data_loaders[0], data_loaders[1])
        cur_ignore_dim = ignore_dims[i] if i in ignore_dims else None
        ret = trainer.test(data_loaders[-1], cur_ignore_dim, data_args.select_pos)
        results.append(ret)

    results = np.concatenate(results, axis=1)
    trainer.show_metric_results(*results, prefix='Average')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/star.yaml')
    parser.add_argument('--data_name', type=str, default='SWaT', help='Data set to train.')
    parser.add_argument('--test', action='store_true', help='Test mode.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    args = parser.parse_args()

    configs = YAMLParser(args.config_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_args = getattr(configs, args.data_name, None)
    if data_args is None:
        raise KeyError(f'{args.data_name} not found in configs.')
    if hasattr(data_args, 'range'):
        train_multi_entity(configs, args.data_name, args.test)
    else:
        train_single_entity(configs, args.data_name, args.test)
