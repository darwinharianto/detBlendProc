
from detectron2.config import LazyConfig
from detectron2 import model_zoo
from detectron2.config import LazyConfig, instantiate
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
import omegaconf
import collections
from copy import copy
import utils
from det2_utils import load_with_fix_lambda_call_on_conv_norm, change_target_dict_to_value, fix_cfg_settings_for_multiple_or_single_gpu
from det2_utils import get_recursively, find_path


if __name__ == "__main__":

    model_name="new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ.py"
    dump_cfg_path = f'./test_conf.yaml'

    cfg = LazyConfig.load(model_zoo.get_config_file(model_name))

    result = []
    path = []

    a = get_recursively(cfg, "norm")
    find_path(cfg, "norm", result, path, i=None)
    print(result)
    print(a)
    cfg = change_target_dict_to_value(cfg, 'conv_norm', None)

    LazyConfig.save(cfg, dump_cfg_path)
    
    cfg = load_with_fix_lambda_call_on_conv_norm(dump_cfg_path)
    cfg = fix_cfg_settings_for_multiple_or_single_gpu(cfg)

    # register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
    # print(cfg.model.backbone.norm)
    model = instantiate(cfg.model)