import abc
from base import metrics


class ADTrainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @staticmethod
    def get_metric_results(pred, true, metric=('precision', 'recall', 'f1')):
        results = [getattr(metrics, f'get_{m}')(pred, true) for m in metric]
        return results
