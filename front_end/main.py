import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import random
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import time
import yaml
import subprocess
from glob import glob
import datetime
import albumentations as A
import cv2
import argparse
from utils import get_images_list, get_placeholder_params, load_augmentations_config, fill_placeholders, fig2img, get_current_time_down_to_microsec
from visuals import show_credentials, show_docstring, show_random_params, show_transform_control, get_transformations_params, get_transformations_params_custom
from control import select_checkbox, select_min_max, select_num_interval, select_radio, select_RGB, select_several_nums, replace_none, select_image, select_transformations
from numpy import asarray


def get_random_image_from_coco(json_annot):
    
    dirname = os.path.dirname(json_annot)
    annFile = json_annot

    coco=COCO(annFile)


    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=nms)
    imgIds = coco.getImgIds(catIds=catIds )
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

    return io.imread(os.path.join(dirname, img['file_name']))

def get_ann_details_from_dataset(json_annot):

    annFile = json_annot
    
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    return cats

def get_random_image_from_folder(folder_path):

    image_list = [image for image in os.listdir(folder_path) if (image.endswith('.jpeg') or image.endswith('.png') or image.endswith('.jpg'))]

    picked_image = image_list[random.randint(0, len(image_list)-1)]
    image = io.imread(os.path.join(folder_path, picked_image) )
    
    return image

def predict_image_show_annot(image, predictor, metadata=None):
    from detectron2.utils.visualizer import Visualizer

    if image.shape[2] == 4:
        src_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        src_image = image
    outputs = predictor(src_image)
    v = Visualizer(src_image[:, :, ::-1],
                metadata=metadata, 
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_res = out.get_image()[:, :, ::-1]
    return image_res

def register_detectron_data(cfg):
    try:
        DatasetCatalog.remove(cfg.TRAIN_MODEL_NAME)
        MetadataCatalog.remove(cfg.TRAIN_MODEL_NAME)
    except KeyError:
        print("Data does not exist, nothing removed") 
    register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
    metadata_catalog = MetadataCatalog.get(cfg.TRAIN_MODEL_NAME)
    dataset_catalog = DatasetCatalog.get(cfg.TRAIN_MODEL_NAME)
    
    with open(cfg.ANNOT_PATH, 'r') as f:
        asd = json.loads (f.read())
    if 'keypoints' in asd["categories"][0]:
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_names = asd["categories"][0]['keypoints']
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_flip_map = []

    return dataset_catalog, metadata_catalog

def dataset_sidebar():
    # Sidebar selection
    dataset_path = './dataset'
    if len(os.listdir(dataset_path)) == 0:
        st.error('No dataset under dataset folder found')
        st.stop()

    dataset = st.sidebar.selectbox(
        'Training Dataset',
        [item for item in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, item))]
    )
    selected_dataset_path = os.path.join(dataset_path, dataset)

    json_annot = st.sidebar.selectbox(
        'Annotation',
        [item for item in os.listdir(selected_dataset_path) if item.endswith('.json')]
    )

    selected_annot_path = os.path.join(selected_dataset_path, json_annot)
    return selected_dataset_path, selected_annot_path

def image_from_mapper(cfg, dataset, metadata):
    from kkimgaug.lib.aug_det2 import Mapper
    # mapper set
    if cfg.ALBUMENTATION_AUG_PATH is not None:
        mapper = Mapper(cfg, True, config=cfg.ALBUMENTATION_AUG_PATH, is_config_type_official=True)
    else:
        from detectron2.data.dataset_mapper import DatasetMapper
        mapper = DatasetMapper(cfg)

    mapper_data = mapper(dataset)
    mapper_img = mapper_data["image"].permute(1,2,0).numpy()[:, :, ::-1]
    visualizer = Visualizer(mapper_img, metadata=metadata, scale=1)
    mapper_data['annotations'] = [{} for _ in range(len(dataset['annotations']))]

    # convert from instance to annotation
    for field in mapper_data['instances'].get_fields():
        if 'box' in field:
            for instance, bbox in enumerate(mapper_data['instances'].get(field)):
                mapper_data['annotations'][instance]['bbox'] = bbox.numpy().tolist()
                mapper_data['annotations'][instance]['bbox_mode'] = 0
        elif 'keypoint' in field:
            for instance, keypoints in enumerate(mapper_data['instances'].get(field).tensor):
                mapper_data['annotations'][instance]['keypoints'] = keypoints.numpy().tolist()
        elif 'masks' in field:
            for instance, segs in enumerate(mapper_data['instances'].get(field)):
                mapper_data['annotations'][instance]['segmentation'] = segs
        elif 'class' in field:
            for instance, cat_id in enumerate(mapper_data['instances'].get(field)):
                mapper_data['annotations'][instance]['category_id'] = cat_id
    
    out_mapper_annot = visualizer.draw_dataset_dict(mapper_data)

    return mapper_img, out_mapper_annot.get_image()

def training_mode():
    import torch
    import detectron2
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.config import CfgNode as CN
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog

    from detectron2.modeling.roi_heads import keypoint_head
    from detectron2.modeling.roi_heads import mask_head
    from detectron2.modeling.roi_heads import box_head
    from detectron2.modeling.roi_heads import roi_heads

    

    model_list = [model for model in model_zoo.model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX]

    except_kpt = ['ROI_KEYPOINT_HEAD_REGISTRY', 'build_keypoint_head', 'BaseKeypointRCNNHead']
    keypoint_head_list = [kpt_heads for kpt_heads in keypoint_head.__all__ if kpt_heads not in except_kpt]

    except_box = ['ROI_BOX_HEAD_REGISTRY', 'build_box_head', 'BaseBoxRCNNHead']
    box_head_list = [box_heads for box_heads in box_head.__all__ if box_heads not in except_box]
    # default value is empty
    box_head_list.append("")

    except_mask = ['ROI_MASK_HEAD_REGISTRY', 'build_mask_head', 'BaseMaskRCNNHead']
    mask_head_list = [mask_heads for mask_heads in mask_head.__all__ if mask_heads not in except_mask]

    except_roi = []
    roi_head_list = [roi_head for roi_head in roi_heads.ROI_HEADS_REGISTRY._obj_map if roi_head not in except_roi]


    cfg = get_cfg()

    st.title('TRAIN DATASET')


    # Sidebar selection
    selected_dataset_path, selected_annot_path = dataset_sidebar()
    aug_path = './augmentations'

    augmentation_cfg = st.sidebar.selectbox(
        'Augmentation',
        [(item if 'augmentations_src.json' not in item else 'None')  for item in os.listdir(aug_path) if (item.endswith('.json') and ~('src' in item))]
    )
    
    if augmentation_cfg != "None":
        selected_aug_path = os.path.join(aug_path, augmentation_cfg)
        loaded_transform = A.load(selected_aug_path)
        cfg.ALBUMENTATION_AUG_PATH = selected_aug_path
    else:
        selected_aug_path = None
        loaded_transform = None
        cfg.ALBUMENTATION_AUG_PATH = None



    mode = st.sidebar.selectbox(
        'Detection mode',
        ['Image', 'Text - Not implemented']
    )

    cfg.ANNOT_PATH = selected_annot_path
    cfg.IMAGE_PATH = selected_dataset_path
    cats = get_ann_details_from_dataset(json_annot=selected_annot_path)
    
    model = st.sidebar.selectbox(
        'Desired model',
        model_list
    )
    st.text(f'Selected model: {model}')
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    train_model_name = st.text_input('Unique model name', f'generic_{model}')
    cfg.TRAIN_MODEL_NAME = train_model_name

    cfg.DATASETS.TRAIN = (train_model_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset

    output_dir =  st.sidebar.text_input(
        'Output Directory',
        value= 'result'
    )
    cfg.OUTPUT_DIR = output_dir

    min_size_train = st.sidebar.text_input(
        'Min input size train',
        value=(800,)
    )

    if type(eval(min_size_train)).__name__ == 'tuple':
        print("Fine")
    else:
        st.sidebar.error('Input Tuple of integer')
    max_size_train = st.sidebar.text_input(
        'Max input size train',
        value=1333
    )

    min_size_test = st.sidebar.text_input(
        'Min input size test',
        value=800
    )
    max_size_test = st.sidebar.text_input(
        'Max input size test',
        value=1333
    )

    cfg.INPUT.MIN_SIZE_TRAIN = eval(min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = int(max_size_train)
    cfg.INPUT.MIN_SIZE_TEST = int(min_size_test)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size_test)

    max_iter =  st.sidebar.text_input(
        'Maximum iteration',
        value=1000
    )
    cfg.SOLVER.MAX_ITER = (
        int(max_iter)
    )  # 300 iterations seems good enough, but you can certainly train longer

    # load cfg

    # model related settings
    roi_head = st.sidebar.selectbox(
        'ROI head',
        roi_head_list,
        index=roi_head_list.index(cfg.MODEL.ROI_HEADS.NAME)
    )
    cfg.MODEL.ROI_HEADS.NAME = roi_head

    num_classes = st.sidebar.text_input(
        'Total classes',
        value=len(cats)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    box_head = st.sidebar.selectbox(
        'Box head',
        box_head_list,
        index=box_head_list.index(cfg.MODEL.ROI_BOX_HEAD.NAME)
    )


    if cfg.MODEL.MASK_ON:
        mask_head = st.sidebar.selectbox(
            'Mask head',
            mask_head_list, 
            index=mask_head_list.index(cfg.MODEL.ROI_MASK_HEAD.NAME)
        )
        cfg.MODEL.ROI_MASK_HEAD.NAME = mask_head

    if cfg.MODEL.KEYPOINT_ON:
        kpt_head = st.sidebar.selectbox(
            'Keypoint head',
            keypoint_head_list,
            index=keypoint_head_list.index(cfg.MODEL.ROI_KEYPOINT_HEAD.NAME)
        )
        cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = kpt_head
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(cats[0]['keypoints'])

    num_worker = st.sidebar.text_input(
        'Number of worker',
        value=2
    )
    cfg.DATALOADER.NUM_WORKERS = int(num_worker)

    ims_per_batch = st.sidebar.text_input(
        'IMS per batch',
        value=2
    )
    cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)

    base_lr =  st.sidebar.text_input(
        'Base Learning Rate',
        value=0.002
    )
    cfg.SOLVER.BASE_LR = float(base_lr)

    checkpoint_period = st.sidebar.text_input(
        'Checkpoint period',
        value = 1000   
    )
    cfg.SOLVER.CHECKPOINT_PERIOD = int(checkpoint_period)

    vis_period =  st.sidebar.text_input(
        'Visualize dataset on train period',
        value = 0
    )
    cfg.VIS_PERIOD = int(vis_period)

    batch_size_per_image =  st.sidebar.text_input(
        'Batch size per image',
        value=128
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        int(batch_size_per_image)
    )  # faster, and good enough for this toy dataset
    cfg.INPUT.RANDOM_FLIP = "none"

    solver_steps = st.sidebar.text_input(
        'Solver steps',
        value = cfg.SOLVER.STEPS
    )

    cfg.SOLVER.STEPS = solver_steps

    # Data Visualization
    # register metadata, then load dataset and metadata
    dataset_catalog, metadata = register_detectron_data(cfg)
    
    # original set
    dataset = dataset_catalog[0]

    ori_img = asarray(Image.open(dataset['file_name']))

    visualizer = Visualizer(ori_img, metadata=metadata, scale=1)
    out_ori = visualizer.draw_dataset_dict(dataset)
    ori_with_annot = out_ori.get_image()


    ori_col1, ori_col2 = st.beta_columns(2)
    map_col1, map_col2 = st.beta_columns(2)

    ori_col1.header("Original")
    ori_img_frame = ori_col1.image(ori_img, use_column_width=True) # normal image
    ori_col2.header("Original with annot")
    ori_img_annot_frame = ori_col2.image(ori_with_annot, use_column_width=True) # normal image


    mapper_img, mapper_img_annot = image_from_mapper(cfg, dataset, metadata)
    map_col1.header("Model input")
    mapper_frame = map_col1.image(mapper_img, use_column_width=True) # mapper image
    map_col2.header("Model input annots")
    mapper_annot_frame = map_col2.image(mapper_img_annot, use_column_width=True) #mapper image with annot 


    animate = st.button('Animate')

    if animate:
        stop = st.button('Stop Animate')
        dataset_iterator = 0
        while not stop:

            dataset_iterator+=1

            dataset = dataset_catalog[dataset_iterator%len(dataset_catalog)]

            # normal image
            ori_img = asarray(Image.open(dataset['file_name']))
            visualizer = Visualizer(ori_img, metadata=metadata, scale=1)
            out_ori = visualizer.draw_dataset_dict(dataset)
            ori_with_annot = out_ori.get_image()


            ori_img_frame.image(ori_img)
            ori_img_annot_frame.image(ori_with_annot)

            # augmented image
            mapper_img, mapper_img_annot = image_from_mapper(cfg, dataset, metadata)
            mapper_frame.image(mapper_img, use_column_width=True)
            mapper_annot_frame.image(mapper_img_annot, use_column_width=True)
            
            time.sleep(0.5)

    st.subheader("Config files Summary")
    st.text(f"""
        {cfg}
    """)

    ### train session?

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if st.button('Start train', key=None):
        dump_cfg_path = './detectron2_cfg/cfg_train.yaml'
        with open(dump_cfg_path, 'w') as f:
            f.write(cfg.dump())
        venv_path = os.popen('echo $VIRTUAL_ENV').read().strip()
        screen_name = f'train_model_{get_current_time_down_to_microsec()}'
        process = subprocess.Popen(['screen', '-S', screen_name, '-d', '-m'], stdout=subprocess.PIPE)
        err_code = process.wait()
        
        if err_code == 0:
            # os.system(f"screen -r {screen_name} -X stuff '{venv_path}/bin/python3 ./front_end/train.py {dump_cfg_path}\n'")  
            st.success('Train command success. Check with screen -r')
        else:
            st.warning('Failed, check for error.')
        
    


def inference_mode():
    import cv2
    st.title('INFERENCE DATASET')

    inference_mode = st.sidebar.radio(
        'Inference Type',
        ['From Image Folder', 'Config Dataset']
    )
    
    cfg_path = './detectron2_cfg'
    selected_cfg = st.sidebar.selectbox(
        'Configuration',
        os.listdir(cfg_path)
    )

    cfg = CN(new_allowed=True)
    cfg_default = get_cfg()
    cfg.merge_from_other_cfg(cfg_default)
    cfg.merge_from_file(os.path.join(cfg_path, selected_cfg))

    model_output = cfg.OUTPUT_DIR

    model_list = [model for model in os.listdir(model_output) if model.endswith('.pth')]

    if len(model_list) == 0:
        st.error('No weights found')
        st.stop()

    selected_model_list = st.sidebar.selectbox(
        'Model Weight',
        model_list,
        model_list.index('model_final.pth') if 'model_final.pth' in model_list else 0 
    )

    cfg.MODEL.WEIGHTS = os.path.join(model_output,selected_model_list)

    with open(cfg.ANNOT_PATH, 'r') as f:
            asd = json.loads (f.read())
    if 'keypoints' in asd["categories"][0]:
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_names = asd["categories"][0]['keypoints']
        MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_flip_map = []

    st.text(f'Model name: {cfg.TRAIN_MODEL_NAME}')
    
    if inference_mode == 'From Image Folder':
        
        # select datasetpath and annot path
        selected_dataset_path, selected_annot_path = dataset_sidebar()

        # register metadata, then load dataset and metadata
        try:
            DatasetCatalog.remove(cfg.TRAIN_MODEL_NAME)
        except KeyError:
            print("Data does not exist, nothing removed") 

        # register data based on selected items
        register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, selected_annot_path, selected_dataset_path)
        metadata = MetadataCatalog.get(cfg.TRAIN_MODEL_NAME)
        dataset_catalog = DatasetCatalog.get(cfg.TRAIN_MODEL_NAME)


        image = get_random_image_from_folder(selected_dataset_path)
        if image.shape[2] == 4:
            src_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            src_image = image

        predictor = DefaultPredictor(cfg)
        
        outputs = predictor(src_image)
        
        st.text('INPUT IMAGE')
        source_image = st.image(src_image)

        # Prediction result
        v = Visualizer(src_image[:, :, ::-1],
                    metadata=metadata, 
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.text('PREDICT IMAGE RESULT')
        predict_image = st.image(out.get_image()[:, :, ::-1])

        animate = st.button('Animate')
        if animate:

            my_bar = st.progress(0)
            all_images = os.listdir(selected_dataset_path)
            for i, image_name in enumerate(all_images):
                my_bar.progress(i/len(all_images))
                st.write(os.path.join(selected_dataset_path, image_name))
                if image_name.endswith('.json'):
                    continue

                image = io.imread(os.path.join(selected_dataset_path, image_name))
                image_res = predict_image_show_annot(image=image, predictor=predictor, metadata=metadata)

                source_image.image(image)
                predict_image.image(image_res)
                if i==10:
                    break

        # dump target for evaluate
        dump_path = st.sidebar.text_input('Dump target', "./output_eval/")

        if st.button('Evaluate'):
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data import build_detection_test_loader
            from utils import st_stdout
            
            # register data
            try:
                DatasetCatalog.remove('eval_item')
            except KeyError:
                print("Data does not exist, nothing removed") 
                
            register_coco_instances('eval_item', {}, selected_annot_path, selected_dataset_path)

            evaluator = COCOEvaluator("eval_item", None, False, output_dir=dump_path)

            val_loader = build_detection_test_loader(cfg, "eval_item")

            import logging
            from utils import MyHandler
            logger = logging.getLogger('detectron2.evaluation.evaluator')
            logger.addHandler(MyHandler())   # or: logger.handlers = [MyHandler()]

            with st_stdout("code"):
                # logger.info("asd")
                print(inference_on_dataset(predictor.model, val_loader, evaluator))
            
            st.text('Outputs AP result')

    elif inference_mode == 'Config Dataset':

        # register metadata, then load dataset and metadata
        try:
            DatasetCatalog.remove(cfg.TRAIN_MODEL_NAME)
        except KeyError:
            print("Data does not exist, nothing removed") 
        register_coco_instances(cfg.TRAIN_MODEL_NAME, {}, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
        metadata = MetadataCatalog.get(cfg.TRAIN_MODEL_NAME)
        dataset_catalog = DatasetCatalog.get(cfg.TRAIN_MODEL_NAME)
        
        with open(cfg.ANNOT_PATH, 'r') as f:
                asd = json.loads (f.read())
        if 'keypoints' in asd["categories"][0]:
            MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_names = asd["categories"][0]['keypoints']
            MetadataCatalog.get(cfg.TRAIN_MODEL_NAME).keypoint_flip_map = []


        # This part shows source image
        st.subheader('Input Image')
        image = io.imread(dataset_catalog[0]['file_name'])
        st.image(image)

        # this part shows annotated image
        st.subheader('Ground Truth Image')
        img = cv2.imread(dataset_catalog[0]["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(dataset_catalog[0])
        st.image(out.get_image()[:, :, ::-1])

        # this part shows inference result
        predictor = DefaultPredictor(cfg)
        
        image_res = predict_image_show_annot(image=image, predictor=predictor, metadata=metadata)
        st.subheader('Predict Result')
        st.image(image_res)
    
    torch.cuda.empty_cache()


def data_gen_mode():
    st.title('DATA GENERATOR')

    blenderProcFolder = './BlenderProc'
    config_dir_location = 'data_generation/config'

    blend_cfg_file = st.sidebar.selectbox(
        'BlenderProc Settings',
        os.listdir(os.path.join(config_dir_location))
    )

    blend_cfg_file_path = os.path.abspath(os.path.join(config_dir_location, blend_cfg_file))
    
    with open(blend_cfg_file_path, 'r') as file:
        blend_cfg = yaml.load(file, Loader=yaml.FullLoader)
    

    st.subheader('Blend Configs')
    cfg_text = st.text_area(
        'Loaded config',
        f"""{yaml.dump(blend_cfg)}""",
        height=1500
    )
    with open(blend_cfg_file_path, 'w') as outfile:
        yaml.dump(yaml.load(cfg_text), outfile, default_flow_style=False)

    if st.button('Generate Preview'):


        process = subprocess.Popen(['timeout' ,'20', 'python', '../BlenderProc/run.py', blend_cfg_file_path], cwd=os.path.abspath(blenderProcFolder), stdout=subprocess.PIPE)
        
        with st.spinner(text='Generating Data...'):
            err_code = process.wait()
        if err_code == 0:
            st.success('Done!')
            st.subheader('Image Preview')
            
            output_dir = [module for module in blend_cfg["modules"] if module["module"] == "main.Initializer"][0]["config"]["global"]["output_dir"]
            
            coco_data_path = os.path.join(output_dir, 'coco_data', 'coco_annotations.json')
            random_image = get_random_image_from_coco(coco_data_path)
            st.image(random_image)
        else:
            st.warning('Generating Process Took Too Long, Open terminal, type screen -r')
            venv_path = os.popen('echo $VIRTUAL_ENV').read().strip()
            timer = get_current_time_down_to_microsec()
            screen_name = f'data_generate_{timer}'

            process = subprocess.Popen(['screen', '-S', screen_name, '-d', '-m'], cwd=os.path.abspath(blenderProcFolder), stdout=subprocess.PIPE)
            err_code = process.wait()
            
            if err_code == 0:
                os.system(f"screen -r {screen_name} -X stuff '{venv_path}/bin/python3 ../BlenderProc/run.py {blend_cfg_file_path}\n'")          

        



def aug_gen_mode():
    st.title('AUGMENTATION GENERATOR')

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
        
    if uploaded_file is None:
        st.title("There is no Image")
    else:
        # select interface type
        interface_type = "Custom"

        json_file_name_input = st.sidebar.text_input("Insert Json File Name", "aug_file")  #text_area same format
        json_file_name = os.path.join("augmentations",f"{json_file_name_input}"+'.json')                       
        
        # select image
        image = asarray(Image.open(uploaded_file))
        # image was loaded successfully
        placeholder_params = get_placeholder_params(image)
    
        # load the config
        augmentations = load_augmentations_config(
            placeholder_params, "augmentations/augmentations_src.json"
        )

        # get the list of transformations names
        checkbox =  st.sidebar.checkbox("Group Transformation (Experimental)",False)
        transform_names = select_transformations(augmentations, interface_type, checkbox)
        
        if checkbox:
            transforms = get_transformations_params_custom(transform_names, augmentations)

            st.text('Source Image')
            st.image(image)
            
            st.text('Augmented Image')
            transform = A.ReplayCompose(transforms)
            for group in transform:
                group.always_apply = True
                for aug in group:
                    aug.always_apply = True
            
            st.image(transform(image=image)['image'])
            
            for group in transform:
                group.always_apply = False
                for aug in group:
                    aug.always_apply = False
            

        else:
            transforms = get_transformations_params(transform_names, augmentations)

            st.text('Source Image')
            st.image(image)
            
            st.text('Augmented Image')
            transform = A.ReplayCompose(transforms)
            for aug in transform:
                aug.always_apply = True
            
            
            st.image(transform(image=image)['image'])
            
            for aug in transform:
                aug.always_apply = False

        st.write(transform._to_dict())
        
        if st.sidebar.button("Save"):
            coco_transform = A.ReplayCompose(transform.transforms,
            bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_bbox']),
            keypoint_params=A.KeypointParams(format='xy', label_fields=['label_kpt'], remove_invisible=True, angle_in_degrees=True)
            )
            A.save(coco_transform, json_file_name)
            st.sidebar.success("File has been saved.")

def proto_mode():
    st.header('Testing scripts and functions page')


    start = st.button('start')

    if start:
        
        stop = st.button('stop')        
        while not stop:
            st.write(1)
            time.sleep(0.5)

genre = st.sidebar.radio(
    "DESIRED MODE",
    ('Data Generator', 'Augmentation Generator', 'Train', 'Inference', 'Prototype'))

if genre == 'Train':
    training_mode()
elif genre == 'Inference':
    inference_mode()
elif genre == 'Data Generator':
    data_gen_mode()
elif genre == 'Augmentation Generator':
    aug_gen_mode()
elif genre == 'Prototype':
    proto_mode()