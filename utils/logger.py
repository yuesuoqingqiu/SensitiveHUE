import logging
from collections import OrderedDict
from logging.handlers import RotatingFileHandler


def singleton(cls, *args, **kwargs):
    _instances = dict()

    def _singleton():
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return _singleton


@singleton
class Logger:
    def __init__(self):
        formatter = logging.Formatter('%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.__formatter = formatter
        self.__console = stream_handler
        self.__loggers = OrderedDict()

    def __get_file_handler(self, file_path: str):
        file_handler = RotatingFileHandler(file_path, maxBytes=10 * 1024 ** 2, backupCount=3)
        file_handler.setFormatter(self.__formatter)
        file_handler.setLevel(logging.INFO)
        return file_handler

    def get_logger(self, name=None, file_path=None):
        if name is None:
            assert len(self.__loggers) > 0, 'No loggers is defined!'
            logger = next(iter(self.__loggers.values()))
        elif name not in self.__loggers:
            assert file_path is not None, 'file_path should be specified!'
            file_handler = self.__get_file_handler(file_path)
            logger = logging.getLogger(name)
            logger.setLevel(level=logging.INFO)
            logger.addHandler(self.__console)
            logger.addHandler(file_handler)
            self.__loggers[name] = logger
        else:
            logger = self.__loggers[name]
        return logger
