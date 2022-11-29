import os
import yaml
from easydict import EasyDict as edict

class Config:
    def __init__(self):
        prefix = 'src'
        self.source = f'{prefix}/sample.mp4'
        self.distortion_data = f'{prefix}/cam_calib_v2.pkl'
        self.camera_space = [[209, 709], [303, 709], [313, 810], [203, 811]]
        self.world_space = [[642, 659], [1174, 496], [1607, 671], [977, 1147]]
        self.model_path = f'{prefix}/yolov7-navibox-best-221121.pt'
        self.map = f'{prefix}/sodam_seg.png'


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)
