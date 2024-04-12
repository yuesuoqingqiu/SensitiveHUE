import torch
import argparse
import rec_strategy as rs
import torch.optim as optim
from config.parser import YAMLParser
from main import get_data_loaders
from sklearn.preprocessing import StandardScaler


def train_single_entity(args, mask_mode: str, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    scaler = StandardScaler()
    data_loaders = get_data_loaders(data_args, args, scaler)
    # ------------------------ Trainer Setting ----------------------------
    model = rs.Transformer(
        data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
        data_args.dim_hidden_fc, data_args.encode_layer_num, 0.1, mask_mode=='normal'
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    temp_mask_mode = 'none' if mask_mode == 'normal' else mask_mode
    mask = rs.TSMask(args.mask_ratio, args.mask_val, temp_mask_mode)
    trainer = rs.Trainer(
        model, optimizer, mask, args.max_epoch, data_args.model_save_dir,
        scheduler, use_prob=False, model_save_suffix=mask_mode
    )

    if not only_test:
        trainer.train(data_loaders[0], data_loaders[1])

    ignore_dims = None
    if hasattr(data_args, 'ignore_dims'):
        ignore_dims = data_args.ignore_dims
    trainer.test(data_loaders[-1], ignore_dims, data_args.topk)
    trainer.test_rec_temporal(data_loaders[1])
    trainer.test_rec_spatial(data_loaders[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/star_compression.yaml')
    parser.add_argument('--mask_mode', type=str, default='none')
    parser.add_argument('--data_name', type=str, default='SWaT', help='Data set to train.')
    parser.add_argument('--test', action='store_true', help='Test mode.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    args = parser.parse_args()

    configs = YAMLParser(args.config_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_single_entity(configs, args.mask_mode, args.data_name, args.test)
