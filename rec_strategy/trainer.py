import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .model import TSMask
from scipy.stats import iqr
from torch.optim import Optimizer
from collections import OrderedDict
from utils import EarlyStop, Logger
from base import ADTrainer, get_best_threshold, adjust_predicts


class Trainer(ADTrainer):
    def __init__(self, model, optimizer, mask_module: TSMask, max_epoch, model_save_dir,
                 scheduler=None, use_prob=True, model_save_suffix=''):
        self.model: nn.Module = model
        self.optimizer: Optimizer = optimizer
        self.max_epoch: int = max_epoch
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        self.early_stop = EarlyStop(tol_num=10, min_is_best=True)
        self.metrics = ('precision', 'recall', 'f1')
        self._mse_loss = nn.MSELoss(reduction='mean')
        self._mask_module = mask_module
        self._use_prob = use_prob

        mode = 'none' if mask_module.mode == 'none' else 'mid'
        # mode = 'mid'
        self._mask_module_test = TSMask(
            self._mask_module.mask_ratio, self._mask_module.masked_val, mode=mode
        )

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.logger = Logger().get_logger(
            str(model), os.path.join(model_save_dir, f'log_{model}.txt'))

        model_save_name = f'{model}_{model_save_suffix}'
        if use_prob:
            model_save_name += '_prob'
        self.model_save_path: str = os.path.join(model_save_dir, model_save_name + '.pkl')
        self.loss_keys = ('rec', 'prob')

    def show_metric_results(self, preds, preds_adjust, labels, prefix='Test'):
        temp_prefix = ('Before PA', 'After PA')
        for i, pred_result in enumerate((preds, preds_adjust)):
            test_results = self.get_metric_results(pred_result, labels, self.metrics)
            log_info = f'{prefix}\t{temp_prefix[i]}\t'
            for metric_name, ret in zip(self.metrics, test_results):
                log_info += '{}:\t{:.4f}\t'.format(metric_name, ret)
            self.logger.info(log_info)
    
    def loss_func(self, x_hat, x, log_var_recip, with_weight=True):
        rec_loss = self._mse_loss(x_hat, x)
        sigma_loss = rec_loss * log_var_recip.exp() - log_var_recip
        if with_weight:
            var = (-log_var_recip).exp().detach()
            loss = (var * sigma_loss / var.mean(dim=(0, 1))).mean()
            # loss = (var * sigma_loss).mean()
        else:
            loss = sigma_loss.mean()
        return rec_loss.mean(), loss

    def train_one_epoch(self, data_loader):
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self._mask_module.apply(self.model, x)

            rec_loss, prob_loss = self._mask_module.apply_inverse(self.loss_func, rec, x, log_var_recip)
            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

            self.optimizer.zero_grad()
            loss = prob_loss if self._use_prob else rec_loss
            loss.backward()
            self.optimizer.step()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        func = lambda x_hat, x, sigma: self.loss_func(x_hat, x, sigma, False)
        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self._mask_module.apply(self.model, x)

            rec_loss, prob_loss = self._mask_module.apply_inverse(func, rec, x, log_var_recip)
            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict
    
    @torch.no_grad()
    def test_rec_temporal(self, data_loader):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        rec_deviations, rec_errors_ori, rec_errors_sp = [], [], []
        mask = None

        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            if mask is None:
                mask = torch.logical_not(torch.eye(x.size(1), dtype=torch.bool, device=self.device))
            
            x_rec, _ = self._mask_module_test.apply(self.model, x)
            x_rec_sp, _ = self._mask_module_test.apply(self.model, x, mask=mask)
            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_rec, x
            ).mean(dim=(1, 2))
            rec_errors_ori.extend(rec_error.tolist())

            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_rec_sp, x
            ).mean(dim=(1, 2))
            rec_errors_sp.extend(rec_error.tolist())

            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_rec_sp, x_rec
            ).mean(dim=(1, 2))
            rec_deviations.extend(rec_error.tolist())

        sample_num = len(rec_errors_ori)
        self.logger.info('Temporal Disruption:')
        self.logger.info('Reconstruction Error before disruption: {:.4f}'.format(
            sum(rec_errors_ori) / sample_num))
        self.logger.info('Reconstruction Error after disruption: {:.4f}'.format(
            sum(rec_errors_sp) / sample_num))
        self.logger.info('Reconstruction Deviation: {:.4f}'.format(
            sum(rec_deviations) / sample_num))

    @torch.no_grad()
    def test_rec_spatial(self, data_loader):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        rec_deviations, rec_errors_ori, rec_errors_permute = [], [], []

        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            _, seq_len, var_num = x.size()

            if self._mask_module.mode == 'none':
                x_rec, _ = self.model(x)
            else:
                x_rec, _ = self._mask_module_test.apply(self.model, x)

            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_rec, x
            ).mean(dim=(1, 2))
            rec_errors_ori.extend(rec_error.tolist())

            x_permute_rec = []
            for i in range(var_num):
                x_permute = torch.stack([
                    x[:, torch.randperm(seq_len, device=self.device), j] if j != i else x[:, :, j]
                    for j in range(var_num)
                ], dim=-1)
                if self._mask_module.mode == 'none':
                    x_rec_temp, _ = self.model(x_permute)
                else:
                    x_rec_temp, _ = self._mask_module_test.apply(self.model, x_permute)
                x_permute_rec.append(x_rec_temp[:, :, i])
            x_permute_rec = torch.stack(x_permute_rec, dim=-1)
            
            # 打乱后的重构误差
            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_permute_rec, x
            ).mean(dim=(1, 2))
            rec_errors_permute.extend(rec_error.tolist())

            # 打乱前后的重构偏差
            rec_error = self._mask_module_test.apply_inverse(
                lambda x, y: F.mse_loss(x, y, reduction='none'),
                x_permute_rec, x_rec
            ).mean(dim=(1, 2))
            rec_deviations.extend(rec_error.tolist())

        sample_num = len(rec_errors_ori)
        self.logger.info('Spatial Disruption:')
        self.logger.info('Reconstruction Error before permute: {:.4f}'.format(
            sum(rec_errors_ori) / sample_num))
        self.logger.info('Reconstruction Error after permute: {:.4f}'.format(
            sum(rec_errors_permute) / sample_num))
        self.logger.info('Reconstruction Deviation: {:.4f}'.format(
            sum(rec_deviations) / sample_num))

    def train(self, train_data_loader, eval_data_loader):
        min_loss_valid = float('inf')
        key = self.loss_keys[int(self._use_prob)]

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            loss_dict_train = self.train_one_epoch(train_data_loader)
            loss_dict_val = self.evaluate(eval_data_loader)

            if loss_dict_val[key] < min_loss_valid:
                min_loss_valid = loss_dict_val[key]
                torch.save(self.model.state_dict(), self.model_save_path)

            log_msg = 'Epoch {:02}/{:02}\tTrain:'.format(epoch, self.max_epoch)
            for k, v in loss_dict_train.items():
                log_msg += '\t{}:{:.6f}'.format(k, v)
            log_msg += '\tValid:'
            for k, v in loss_dict_val.items():
                log_msg += '\t{}:{:.6f}'.format(k, v)
            self.logger.info(log_msg)

            if self.early_stop.reach_stop_criteria(loss_dict_val[key]):
                break

            if self.scheduler:
                self.scheduler.step()
        self.logger.info('Training is over...')

    @torch.no_grad()
    def _get_anomaly_score(self, data_loader, load_state=True):
        if load_state:
            self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        errors, labels = [], []
        for x, label in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self._mask_module_test.apply(self.model, x)

            anomaly_score = F.mse_loss(rec, x, reduction='none')
            if self._use_prob:
                anomaly_score = anomaly_score * log_var_recip.exp() - log_var_recip
            
            pos = x.size(1) // 2 if self._mask_module_test.mode != 'tail' else x.size(1) - 1
            errors.append(anomaly_score[:, pos])
            labels.append(label[:, pos])

        errors = torch.cat(errors, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        return errors, labels

    def test(self, data_loader, ignore_dims=None, topk=1):
        errors, labels = self._get_anomaly_score(data_loader, True)
        preds, preds_adjust = self.get_pred_results(errors, labels, ignore_dims, topk)
        self.show_metric_results(preds, preds_adjust, labels)
        return np.stack((preds, preds_adjust, labels), axis=0)

    def get_pred_results(self, errors: np.ndarray, labels: np.ndarray, ignore_dims=None, topk=1):
        # Normalization
        median, iqr_ = np.median(errors, axis=0), iqr(errors, axis=0)
        errors = (errors - median) / (iqr_ + 1e-9)

        # 最大值法
        if ignore_dims:
            errors[:, ignore_dims] = 0
        final_errors = errors.max(axis=1)

        thr, _ = get_best_threshold(final_errors, labels, adjust=False)
        preds = (final_errors >= thr).astype(int)

        thr_adjust, _ = get_best_threshold(final_errors, labels, adjust=True)
        preds_adjust = (final_errors >= thr_adjust).astype(int)
        preds_adjust = adjust_predicts(preds_adjust, labels)

        return preds, preds_adjust
