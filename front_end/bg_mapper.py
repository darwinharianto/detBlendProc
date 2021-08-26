
from PIL import Image
import copy
import numpy as np
import torch
from kkimgaug.lib.aug_det2 import Mapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]


def has_transparency(img):
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False

def convert_PIL_to_numpy(image, format="BGR"):
    import numpy as np
    """
    Convert PIL image to numpy array of target format.
    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image
    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def get_random_bg_image(folder):
    import os
    import random

    choose_random = random.choice(os.listdir(folder))

    return Image.open(os.path.join(folder, choose_random))

def add_bg_image(background_dir, image):

    if background_dir is not None:
        background = get_random_bg_image(folder = background_dir)
    
        # resize the image
        size = image.size[:2]
        background = background.resize(size,Image.ANTIALIAS)
        background.paste(image, (0, 0), image)
        return background
    else:
        return image

# local files
from kkimgaug.lib import BaseCompose
from kkimgaug.util.procs import bgr2rgb, rgb2bgr, mask_from_polygon_to_bool, mask_from_bool_to_polygon, \
    kpt_from_coco_to_xy, to_uint8, restore_kpt_coco_format, get_applied_augmentations, bbox_label_auto, \
    check_coco_annotations, mask_inside_bbox, bbox_compute_from_mask

class BGMapper(Mapper):

    def __init__(self, *args, background_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_dir = background_dir

    def __call__(self, dataset_dict):
        """
        copy "__call__" function of detectron2.data.dataset_mapper.DatasetMapper 
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        ### BG add if RGBA?##
        image = Image.open(dataset_dict["file_name"])
        if has_transparency(image):
            image = convert_PIL_to_numpy( add_bg_image(self.background_dir, image), format=self.image_format)
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        ### my code ##        

        ##### My Code #####
        transformed = self.composer(
            image=image,
            bboxes=[x["bbox"] if x.get("bbox") else [] for x in dataset_dict["annotations"]],
            mask=[x["segmentation"] if x.get("segmentation") else [] for x in dataset_dict["annotations"]],
            keypoints=[x["keypoints"] if x.get("keypoints") else [] for x in dataset_dict["annotations"]],
        )
        image = transformed["image"]
        if "annotations" in dataset_dict:
            if transformed.get("label_bbox") is not None and len(transformed["label_bbox"]) > 0:
                dataset_dict["annotations"] = np.array(dataset_dict["annotations"])[transformed["label_bbox"]].tolist()
            else:
                dataset_dict["annotations"] = []
            for i, dictwk in enumerate(dataset_dict["annotations"]):
                if "bbox" in dictwk:
                    dictwk["bbox"] = transformed["bboxes"][i]
                if "keypoints" in dictwk and len(dictwk["keypoints"]) > 0:
                    dictwk["keypoints"] = transformed["keypoints"][i]
                if "segmentation" in dictwk and len(dictwk["segmentation"]) > 0:
                    dictwk["segmentation"] = transformed["mask"][i]
        ##### My Code #####

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict