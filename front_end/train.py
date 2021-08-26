from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from bg_mapper import BGMapper

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.config import LazyConfig, instantiate
import click
import json
import logging
from PIL import Image


class CustomKKTrainer(DefaultTrainer):

    def build_train_loader(cls, cfg):
        if cfg.ALBUMENTATION_AUG_PATH is not None:
            mapper = BGMapper(cfg, True, background_dir=cfg.BACKGROUND_IMAGE, config=cfg.ALBUMENTATION_AUG_PATH, is_config_type_official=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)



def do_train_lazyconf(cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optim,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=True)
    if checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    
    trainer.train(start_iter, cfg.train.max_iter)


@click.command()
@click.argument('cfg_path')
def hello(cfg_path):

    if cfg_path.endswith('.yaml'):
        cfg = CN(new_allowed=True)
        cfg_default = get_cfg()
        cfg.merge_from_other_cfg(cfg_default)
        cfg.merge_from_file(cfg_path)
        
        register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
        with open(cfg.ANNOT_PATH, 'r') as f:
            annot = json.loads (f.read())
        
        if 'keypoints' in annot["categories"][0]:
            MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_names = annot["categories"][0]['keypoints']
            MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_flip_map = []

        print(cfg)
        trainer = CustomKKTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        del trainer
    else:
        cfg = LazyConfig.load(cfg_path)
        default_setup(cfg, args= {})
        
        do_train_lazyconf(cfg)

 
if __name__ == '__main__':
    hello()