import yaml
from argparse import Namespace


class YAMLParser:
    def __init__(self, file_path: str):
        assert file_path.endswith('yaml')
        with open(file_path, 'r', encoding='utf-8') as fp:
            config_dict = yaml.safe_load(fp)
        
        for k, v in config_dict.items():
            if not isinstance(v, dict):
                setattr(self, k, v)
            else:
                setattr(self, k, Namespace(**v))
    
    def __contains__(self, key: str):
        return key in self.__dict__
