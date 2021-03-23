
import detectron2
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
import click
from kkimgaug.lib.aug_det2 import Mapper
import json


class CustomKKTrainer(DefaultTrainer):

    def build_train_loader(cls, cfg):
        if cfg.ALBUMENTATION_AUG_PATH is not None:
            mapper = Mapper(cfg, True, config=cfg.ALBUMENTATION_AUG_PATH, is_config_type_official=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)




@click.command()
@click.argument('cfg_path')
def hello(cfg_path):

    cfg = CN(new_allowed=True)
    cfg_default = get_cfg()
    cfg.merge_from_other_cfg(cfg_default)
    cfg.merge_from_file(cfg_path)
    
    register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
    with open(cfg.ANNOT_PATH, 'r') as f:
        asd = json.loads (f.read())
    
    if 'keypoints' in asd["categories"][0]:
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_names = asd["categories"][0]['keypoints']
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_flip_map = []

    trainer = CustomKKTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    del trainer


 
if __name__ == '__main__':
    hello()