import streamlit as st
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
from visuals import show_credentials, show_docstring, show_random_params, show_transform_control, get_transformations_params, get_transformations_params_group
from control import select_checkbox, select_min_max, select_num_interval, select_radio, select_RGB, select_several_nums, replace_none, select_image, select_transformations
from numpy import asarray




def send_command_to_screen(screen_name, command):
    process = subprocess.Popen(['screen', '-S', screen_name, '-d', '-m'], stdout=subprocess.PIPE)
    err_code = process.wait()
    
    if err_code == 0:
        os.system(command)  
        st.success('Train command success. Check with screen -r')
    else:
        st.warning('Failed, To train, contact developer.')
    

def save_aug_transform(transform, file_name):
    coco_transform = A.ReplayCompose(transform.transforms,
    bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.1, label_fields=['label_bbox']),
    keypoint_params=A.KeypointParams(format='xy', label_fields=['label_kpt'], remove_invisible=True, angle_in_degrees=True)
    )
    A.save(coco_transform, file_name)

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

def show_ground_truth_image_with_annot(dataset, metadata):
    from detectron2.utils.visualizer import Visualizer

    ori_img = asarray(Image.open(dataset['file_name']))
    visualizer = Visualizer(ori_img, metadata=metadata, scale=1)
    out_ori = visualizer.draw_dataset_dict(dataset)
    ori_with_annot = out_ori.get_image()

    return ori_img, ori_with_annot

def register_detectron_data(cfg):

    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    try:
        DatasetCatalog.remove(cfg.TRAIN_MODEL_NAME)
        MetadataCatalog.remove(cfg.TRAIN_MODEL_NAME)
    except KeyError:
        print("Data does not exist, nothing removed") 
    
    with open(cfg.ANNOT_PATH, 'r') as f:
        annot = json.loads(f.read())

    extra_metadata = {
        "keypoint_names": annot["categories"][0]['keypoints'],
        "keypoint_flip_map": [],
        }  if 'keypoints' in annot["categories"][0] else {}
    
    register_coco_instances(cfg.TRAIN_MODEL_NAME, extra_metadata, cfg.ANNOT_PATH, cfg.IMAGE_PATH)
    metadata_catalog = MetadataCatalog.get(cfg.TRAIN_MODEL_NAME)
    dataset_catalog = DatasetCatalog.get(cfg.TRAIN_MODEL_NAME)
    
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
    from detectron2.utils.visualizer import Visualizer
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


def get_default_head_list():
    
    from detectron2 import model_zoo
    from detectron2.modeling.roi_heads import keypoint_head
    from detectron2.modeling.roi_heads import mask_head
    from detectron2.modeling.roi_heads import box_head
    from detectron2.modeling.roi_heads import roi_heads

    # detectron2 model zoo list
    model_list = [model for model in model_zoo.model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX]

    # detectron2 kpt list
    except_kpt = ['ROI_KEYPOINT_HEAD_REGISTRY', 'build_keypoint_head', 'BaseKeypointRCNNHead']
    keypoint_head_list = [kpt_heads for kpt_heads in keypoint_head.__all__ if kpt_heads not in except_kpt]

    # detectron2 box head list
    except_box = ['ROI_BOX_HEAD_REGISTRY', 'build_box_head', 'BaseBoxRCNNHead']
    box_head_list = [box_heads for box_heads in box_head.__all__ if box_heads not in except_box]
    # default value is empty
    box_head_list.append("")

    # detectron2 mask head list
    except_mask = ['ROI_MASK_HEAD_REGISTRY', 'build_mask_head', 'BaseMaskRCNNHead']
    mask_head_list = [mask_heads for mask_heads in mask_head.__all__ if mask_heads not in except_mask]

    # detectron2 roi head list
    except_roi = []
    roi_head_list = [roi_head for roi_head in roi_heads.ROI_HEADS_REGISTRY._obj_map if roi_head not in except_roi]
    
    return model_list, roi_head_list, box_head_list, mask_head_list, keypoint_head_list

def training_mode():
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2 import model_zoo



    
    st.title('TRAIN DATASET')

    # Sidebar selection
    selected_dataset_path, selected_annot_path = dataset_sidebar()
    aug_path = './augmentations'

    augmentation_cfg = st.sidebar.selectbox(
        'Augmentation',
        [(item if 'augmentations_src.json' not in item else 'None')  for item in os.listdir(aug_path) if (item.endswith('.json') and ~('src' in item))]
    )
    mode = st.sidebar.selectbox(
        'Detection mode',
        ['Image', 'Text - Not implemented']
    )
    # This should be based with mode, if text or if detectron2
    model_list, roi_head_list, box_head_list, mask_head_list, keypoint_head_list = get_default_head_list()

    model = st.sidebar.selectbox(
        'Desired model',
        model_list
    )
    st.text(f'Selected model: {model}')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cats = get_ann_details_from_dataset(json_annot=selected_annot_path) 

    train_model_name = st.text_input('Unique model name', f'generic_{model}')
    output_dir =  st.sidebar.text_input(
        'Output Directory',
        value= 'result'
    )
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
    max_iter =  st.sidebar.text_input(
        'Maximum iteration',
        value=1000
    )
    # model related settings
    roi_head = st.sidebar.selectbox(
        'ROI head',
        roi_head_list,
        index=roi_head_list.index(cfg.MODEL.ROI_HEADS.NAME)
    )

    num_classes = st.sidebar.text_input(
        'Total classes',
        value=len(cats)
    )
    box_head = st.sidebar.selectbox(
        'Box head',
        box_head_list,
        index=box_head_list.index(cfg.MODEL.ROI_BOX_HEAD.NAME)
    )
    mask_head = st.sidebar.selectbox(
        'Mask head',
        mask_head_list, 
        index=mask_head_list.index(cfg.MODEL.ROI_MASK_HEAD.NAME)
    ) if cfg.MODEL.MASK_ON else mask_head_list.index(cfg.MODEL.ROI_MASK_HEAD.NAME)
    kpt_head = st.sidebar.selectbox(
        'Keypoint head',
        keypoint_head_list,
        index=keypoint_head_list.index(cfg.MODEL.ROI_KEYPOINT_HEAD.NAME)
    ) if cfg.MODEL.KEYPOINT_ON else keypoint_head_list.index(cfg.MODEL.ROI_KEYPOINT_HEAD.NAME)
    num_worker = st.sidebar.text_input(
        'Number of worker',
        value=2
    )
    ims_per_batch = st.sidebar.text_input(
        'IMS per batch',
        value=2
    )
    base_lr =  st.sidebar.text_input(
        'Base Learning Rate',
        value=0.002
    )
    checkpoint_period = st.sidebar.text_input(
        'Checkpoint period',
        value = 1000   
    )
    vis_period =  st.sidebar.text_input(
        'Visualize dataset on train period',
        value = 0
    )
    batch_size_per_image =  st.sidebar.text_input(
        'Batch size per image',
        value=128
    )
    solver_steps = st.sidebar.text_input(
        'Solver steps',
        value = cfg.SOLVER.STEPS
    )

    cfg.ALBUMENTATION_AUG_PATH = None if augmentation_cfg == None else os.path.join(aug_path, augmentation_cfg)
    cfg.ANNOT_PATH = selected_annot_path
    cfg.IMAGE_PATH = selected_dataset_path
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.TRAIN_MODEL_NAME = train_model_name
    cfg.DATASETS.TRAIN = (train_model_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.OUTPUT_DIR = output_dir
    cfg.INPUT.MIN_SIZE_TRAIN = eval(min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = int(max_size_train)
    cfg.INPUT.MIN_SIZE_TEST = int(min_size_test)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size_test)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.MODEL.ROI_HEADS.NAME = roi_head
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_MASK_HEAD.NAME = mask_head
    cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = kpt_head
    # for now only allows one Class of keypoints
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(cats[0]['keypoints']) if cfg.MODEL.KEYPOINT_ON else cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
    cfg.DATALOADER.NUM_WORKERS = int(num_worker)
    cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
    cfg.SOLVER.BASE_LR = float(base_lr)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(checkpoint_period)
    cfg.VIS_PERIOD = int(vis_period)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(batch_size_per_image)
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.SOLVER.STEPS = solver_steps

    # Data Visualization
    # register metadata, then load dataset and metadata
    ori_col1, ori_col2 = st.beta_columns(2)
    map_col1, map_col2 = st.beta_columns(2)
    dataset_catalog, metadata = register_detectron_data(cfg)
    
    # original set
    ori_img, ori_with_annot = show_ground_truth_image_with_annot(dataset_catalog[0], metadata)
    ori_col1.header("Original")
    ori_img_frame = ori_col1.image(ori_img, use_column_width=True) # normal image
    ori_col2.header("Original with annot")
    ori_img_annot_frame = ori_col2.image(ori_with_annot, use_column_width=True) # normal image


    mapper_img, mapper_img_annot = image_from_mapper(cfg, dataset_catalog[0], metadata)
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
            selected_dataset = dataset_catalog[dataset_iterator%len(dataset_catalog)]
            # normal image
            ori_img, ori_with_annot = show_ground_truth_image_with_annot(selected_dataset, metadata)
            ori_img_frame.image(ori_img)
            ori_img_annot_frame.image(ori_with_annot)

            # augmented image
            mapper_img, mapper_img_annot = image_from_mapper(cfg, selected_dataset, metadata)
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
        
        dump_cfg_path = f'./detectron2_cfg/cfg_train_{get_current_time_down_to_microsec()}.yaml'
        with open(dump_cfg_path, 'w') as f:
            f.write(cfg.dump())
        venv_path = os.popen('echo $VIRTUAL_ENV').read().strip()

        screen_name = f'train_model_{dump_cfg_path}'
        command = f"screen -r {screen_name} -X stuff '{venv_path}/bin/python3 ./front_end/train.py {dump_cfg_path}\n'"
        send_command_to_screen(screen_name = screen_name, command = command)



def inference_mode():
    import cv2
    from detectron2.config import CfgNode as CN
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg_path = './detectron2_cfg'

    st.title('INFERENCE DATASET')
    inference_mode = st.sidebar.radio(
        'Inference Type',
        ['From Image Folder', 'Config Dataset']
    )
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
    st.text(f'Model name: {cfg.TRAIN_MODEL_NAME}')
    
    if inference_mode == 'From Image Folder':
        
        # select datasetpath and annot path
        selected_dataset_path, _ = dataset_sidebar()
        # dump target for evaluate
        dump_path = st.sidebar.text_input('Dump target', "./output_eval/")

        # register metadata, then load dataset and metadata
        cfg.IMAGE_PATH = selected_dataset_path
        dataset_catalog, metadata = register_detectron_data(cfg)

        image = get_random_image_from_folder(selected_dataset_path)
        predictor = DefaultPredictor(cfg)
        image_res = predict_image_show_annot(image=image, predictor=predictor, metadata=metadata)
        
        st.text('INPUT IMAGE')
        source_image = st.image(image)

        st.text('PREDICT IMAGE RESULT')
        predict_image = st.image(image_res)

        animate = st.button('Animate')
        if animate:

            stop = st.button('Stop Animate')
            my_bar = st.progress(0)
            all_images = [images for images in os.listdir(selected_dataset_path) if not images.endswith('.json')]
            while (not stop):
                for i, image_name in enumerate(all_images):
                    my_bar.progress((i+1)/len(all_images))
                    image = io.imread(os.path.join(selected_dataset_path, image_name))
                    image_res = predict_image_show_annot(image=image, predictor=predictor, metadata=metadata)
                    source_image.image(image)
                    predict_image.image(image_res)


        if st.button('Evaluate'):
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data import build_detection_test_loader
            from utils import st_stdout
            
            # register data
            dataset_catalog, metadata = register_detectron_data(cfg)
                
            evaluator = COCOEvaluator(cfg.TRAIN_MODEL_NAME, None, False, output_dir=dump_path)
            val_loader = build_detection_test_loader(cfg, cfg.TRAIN_MODEL_NAME)
            import logging
            from utils import MyHandler
            logger = logging.getLogger('detectron2.evaluation.evaluator')
            logger.handlers = [MyHandler()]

            with st_stdout("code"):
                asd = inference_on_dataset(predictor.model, val_loader, evaluator)
                print(asd)
            

    elif inference_mode == 'Config Dataset':

        # register metadata, then load dataset and metadata
        dataset_catalog, metadata = register_detectron_data(cfg)

        # This part shows source image
        st.subheader('Input Image')
        image = io.imread(dataset_catalog[0]['file_name'])
        st.image(image)

        img1, img2 = st.beta_columns(2)
        
        # this part shows annotated image
        img1.header('Ground Truth Image')
        _, out_img = show_ground_truth_image_with_annot(dataset_catalog[0], metadata)
        img1.image(out_img, use_column_width=True)

        # this part shows inference result
        predictor = DefaultPredictor(cfg)
        
        image_res = predict_image_show_annot(image=image, predictor=predictor, metadata=metadata)
        img2.header('Predict Result')
        img2.image(image_res, use_column_width=True)
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        print('torch is not installed')

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
            st.warning('Generating process took too long. To see progress, open terminal, type screen -r')
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

        # select image
        image = asarray(Image.open(uploaded_file))
        # load the config
        augmentations = load_augmentations_config(
            get_placeholder_params(image), "augmentations/augmentations_src.json"
        )

        json_file_name_input = st.sidebar.text_input("Insert Json File Name", "aug_file")  #text_area same format
        json_file_name = os.path.join("augmentations",f"{json_file_name_input}"+'.json')     
        
        # get the list of transformations names
        checkbox = st.sidebar.checkbox("Group Transformation (Experimental)",False)
        transform_names = select_transformations(augmentations, checkbox)
        
        if checkbox:
            transforms = get_transformations_params_group(transform_names, augmentations)
            transform = A.ReplayCompose(transforms)
        else:
            transforms = get_transformations_params(transform_names, augmentations)
            transform = A.ReplayCompose(transforms)

        import itertools
        from collections.abc import Iterable
        if isinstance(transform, Iterable):
            for item in itertools.chain.from_iterable(transform):
                if issubclass(item.__class__, A.BasicTransform):
                    item.always_apply = True
        else:
            for item in transform:
                if issubclass(item.__class__, A.BasicTransform):
                    item.always_apply = True
                
        st.text('Source Image')
        st.image(image)
        st.text('Augmented Image')
        st.image(transform(image=image)['image'])


        if isinstance(transform, Iterable):
            for item in itertools.chain.from_iterable(transform):
                if issubclass(item.__class__, A.BasicTransform):
                    item.always_apply = False
        else:
            for item in transform:
                if issubclass(item.__class__, A.BasicTransform):
                    item.always_apply = False
        
        st.write(transform._to_dict())
        
        if st.sidebar.button("Save"):
            save_aug_transform(transform, json_file_name)
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