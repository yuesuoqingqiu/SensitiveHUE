import os
import torch
import numpy as np
import torch.nn as nn
from .model import SensitiveHUE
from scipy.stats import iqr
from torch.optim import Optimizer
from collections import OrderedDict
from utils import EarlyStop, Logger
from base import ADTrainer, get_best_threshold, adjust_predicts


class Trainer(ADTrainer):
    def __init__(self, model, optimizer, alpha, max_epoch, model_save_dir,
                 scheduler=None, use_prob=True, model_save_suffix=''):
        self.model: SensitiveHUE = model
        self.optimizer: Optimizer = optimizer
        self.alpha: float = alpha
        self.max_epoch: int = max_epoch
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        self.early_stop = EarlyStop(tol_num=10, min_is_best=True)
        self.metrics = ('precision', 'recall', 'f1')
        self._mse_loss = nn.MSELoss(reduction='none')
        self._use_prob = use_prob

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.logger = Logger().get_logger(
            str(model), os.path.join(model_save_dir, f'log_{model}.txt'))

        self.model_save_name = f'{model}{model_save_suffix}.pkl'
        self.model_save_path: str = os.path.join(model_save_dir, self.model_save_name)

        self.loss_keys = ('rec', 'prob')

    def _show_param_nums(self):
        params_num = 0
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            params_num += np.prod(p.size())
        self.logger.info('Trainable params num: {}'.format(params_num))
        # self.logger.info(self.model)

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
            mean_var = var.mean(dim=(0, 1)) ** self.alpha
            loss = (var * sigma_loss / mean_var).mean()
        else:
            loss = sigma_loss.mean()
        return rec_loss.mean(), loss

    def train_one_epoch(self, data_loader):
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self.model(x)
            rec_loss, prob_loss = self.loss_func(rec, x, log_var_recip)

            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

            loss = prob_loss if self._use_prob else rec_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for x in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self.model(x)
            rec_loss, prob_loss = self.loss_func(rec, x, log_var_recip, False)

            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict

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
    def _get_anomaly_score(self, data_loader, load_state=True, select_pos='mid'):
        assert select_pos in ('mid', 'tail')
        if load_state:
            self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        errors, labels = [], []
        for x, label in data_loader:
            x = x.type(torch.float32).to(self.device)
            rec, log_var_recip = self.model(x)

            anomaly_score = self._mse_loss(rec, x)
            if self._use_prob:
                anomaly_score = anomaly_score * log_var_recip.exp() - log_var_recip
            
            pos = x.size(1) // 2 if select_pos == 'mid' else x.size(1) - 1
            errors.append(anomaly_score[:, pos])
            labels.append(label[:, pos])

        errors = torch.cat(errors, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        return errors, labels

    def test(self, data_loader, ignore_dims=None, select_pos='mid'):
        errors, labels = self._get_anomaly_score(data_loader, True, select_pos)
        preds, preds_adjust = self.get_pred_results(errors, labels, ignore_dims)
        self.show_metric_results(preds, preds_adjust, labels)
        return np.stack((preds, preds_adjust, labels), axis=0)

    def get_pred_results(self, errors: np.ndarray, labels: np.ndarray, ignore_dims=None):
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
