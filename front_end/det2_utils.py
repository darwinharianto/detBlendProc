
from detectron2.config import LazyConfig
from detectron2 import model_zoo
from detectron2.config import LazyConfig, instantiate
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
import omegaconf
import collections
from copy import copy
import torch

def get_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    https://stackoverflow.com/questions/14962485/finding-a-key-recursively-in-a-dictionary
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, omegaconf.dictconfig.DictConfig):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found

# i is the index of the list that dict_obj is part of
def find_path(dict_obj, key, result, path, i=None):
    """ 
    https://stackoverflow.com/questions/50486643/get-path-of-parent-keys-and-indices-in-dictionary-of-nested-dictionaries-and-l
    """
    for k,v in dict_obj.items():
        # add key to path
        path.append(k)
        if isinstance(v,dict):
            # continue searching
            find_path(v, key, result, path,i)
        if isinstance(v,omegaconf.dictconfig.DictConfig):
            # continue searching
            find_path(v, key, result, path,i)
        if isinstance(v,list):
            # search through list of dictionaries
            for i,item in enumerate(v):
                # add the index of list that item dict is part of, to path
                path.append(i)
                if isinstance(item,dict):
                    # continue searching in item dict
                    find_path(item, key, result, path,i)
                # if reached here, the last added index was incorrect, so removed
                path.pop()
        if k == key:
            # add path to our result
            result.append(copy(path))
        # remove the key added in the first line
        if path != []:
            path.pop()

# TODO change lambda calling
def load_with_fix_lambda_call_on_conv_norm(cfg_path):
    args = LazyConfig.load(cfg_path)

    # this should be called from .py files, i dont know how
    args.model.roi_heads.box_head.conv_norm = \
        args.model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
                                                                           stats_mode="N")
    return args

def fix_cfg_settings_for_multiple_or_single_gpu(cfg):

    if torch.cuda.device_count() < 2:
        cfg = change_target_dict_to_value(cfg, 'norm', "'BN'")

    return cfg


def change_target_dict_to_value(cfg, target, value):
    # result and path should be outside of the scope of find_path to persist values during recursive calls to the function
    result = []
    path = []
    find_path(cfg, target, result, path)
    for paths in result:
        exec(f"cfg.{'.'.join(paths)} = {value}")

    return cfg
