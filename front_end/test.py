
from detectron2.config import LazyConfig
import os
from detectron2 import model_zoo
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
model_name="new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ.py"

cfg = LazyConfig.load(model_zoo.get_config_file(model_name))

train = {
    "checkpointer": {"period": 200}
}
dump_cfg_path = f'./test_conf.py'


cfg_str = LazyConfig.to_py(train)
LazyConfig.save(cfg, dump_cfg_path)


cfg = LazyConfig.load(model_zoo.get_config_file(model_name))
args = LazyConfig.load(dump_cfg_path)
cfg = LazyConfig.apply_overrides(cfg, args)

# default_setup(cfg, args= {})

model = instantiate(cfg.model)