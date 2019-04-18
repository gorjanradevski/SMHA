from tensorflow.contrib.training import HParams
from ruamel.yaml import YAML


class YParams(HParams):
    def __init__(self, hparams_path: str):
        super().__init__()
        with open(hparams_path) as fp:
            for k, v in YAML().load(fp).items():
                self.add_hparam(k, v)
