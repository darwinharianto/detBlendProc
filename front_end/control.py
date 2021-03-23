import streamlit as st
import cv2
from utils import get_images_list, remove_ifpicked
import numpy as np

def select_num_interval(
    param_name: str, limits_list: list, defaults, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    min_max_interval = st.sidebar.slider(
        "",
        limits_list[0],
        limits_list[1],
        defaults,
        key=hash(param_name + str(n_for_hash)),
    )
    return min_max_interval


def select_several_nums(
    param_name, subparam_names, limits_list, defaults_list, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    result = []
    assert len(limits_list) == len(defaults_list)
    assert len(subparam_names) == len(defaults_list)

    for name, limits, defaults in zip(subparam_names, limits_list, defaults_list):
        result.append(
            st.sidebar.slider(
                name,
                limits[0],
                limits[1],
                defaults,
                key=hash(param_name + name + str(n_for_hash)),
            )
        )
    return tuple(result)


def select_min_max(
    param_name, limits_list, defaults_list, n_for_hash, min_diff=0, **kwargs
):
    assert len(param_name) == 2
    result = list(
        select_num_interval(
            " & ".join(param_name), limits_list, defaults_list, n_for_hash
        )
    )
    if result[1] - result[0] < min_diff:
        diff = min_diff - result[1] + result[0]
        if result[1] + diff <= limits_list[1]:
            result[1] = result[1] + diff
        elif result[0] - diff >= limits_list[0]:
            result[0] = result[0] - diff
        else:
            result = limits_list
    return tuple(result)


def select_RGB(param_name, n_for_hash, **kwargs):
    result = select_several_nums(
        param_name,
        subparam_names=["Red", "Green", "Blue"],
        limits_list=[[0, 255], [0, 255], [0, 255]],
        defaults_list=[0, 0, 0],
        n_for_hash=n_for_hash,
    )
    return tuple(result)


def replace_none(string):
    if string == "None":
        return None
    else:
        return string


def select_radio(param_name, options_list, n_for_hash, **kwargs):
    st.sidebar.subheader(param_name)
    result = st.sidebar.radio("", options_list, key=hash(param_name + str(n_for_hash)))
    return replace_none(result)


def select_checkbox(param_name, defaults, n_for_hash, **kwargs):
    st.sidebar.subheader(param_name)
    result = st.sidebar.checkbox(
        "True", defaults, key=hash(param_name + str(n_for_hash))
    )
    return result

def select_image(path_to_images: str, interface_type: str = "Simple"):
    """ Show interface to choose the image, and load it
    Args:
        path_to_images (dict): path ot folder with images
        interface_type (dict): mode of the interface used
    Returns:
        (status, image)
        status (int):
            0 - if everything is ok
            1 - if there is error during loading of image file
            2 - if user hasn't uploaded photo yet
    """
    image_names_list = get_images_list(path_to_images)
    if len(image_names_list) < 1:
        return 1, 0
    else:
        image_name = st.sidebar.selectbox(
            "Select an image:", image_names_list
        )

        try:
            image = load_image(image_name, path_to_images)
            return 0, image
        except cv2.error:
            return 1, 0
        

def select_transformations(augmentations: dict, interface_type: str, is_group: bool) -> list:
    #Customized
    checkbox = is_group
    all_listed_augmentations = sorted(list(augmentations.keys()))
    all_listed_augmentations = [augs for augs in all_listed_augmentations if augs != "OneOf"]
    transform_names = []
    
    if not checkbox:
        transform_names = [
            st.sidebar.selectbox(
                "Select transformation №1:", all_listed_augmentations
            )
        ]
        while transform_names[-1] != "None":
            transform_names.append(
                st.sidebar.selectbox(
                    f"Select transformation №{len(transform_names) + 1}:",
                    ["None"] + all_listed_augmentations,
                )
            )
        transform_names = transform_names[:-1]
    else:
        transform_group_names = ["OneOf"]

        transform_groups = ([[st.sidebar.selectbox(
            "Select transformation group type G0:", transform_group_names
        ),
        
        st.sidebar.multiselect("List of available augmentation",all_listed_augmentations)
        ]])
        # st.sidebar.slider("Augmentation P", 0.0,1.0, key=999,value= 1.)
        count=0
        while transform_groups[-1][0] != "None":
            # grouping_params(transform_groups[count][0],count = count)
            count+=1
            
            transform_groups.append([
                st.sidebar.selectbox(
                    f"Select transformation group type G{count}:",
                    ["None"] + transform_group_names
                ),
                st.sidebar.multiselect("List of available augmentation",all_listed_augmentations ,key=count),
                # st.sidebar.slider("Augmentation P", 0.0,1.0, key=999-count,value= 1.),
                
            ])
        transform_names=transform_groups[:-1]
    
    
    return transform_names

# dict from param name to function showing this param
param2func = {
    "num_interval": select_num_interval,
    "several_nums": select_several_nums,
    "radio": select_radio,
    "rgb": select_RGB,
    "checkbox": select_checkbox,
    "min_max": select_min_max,
}