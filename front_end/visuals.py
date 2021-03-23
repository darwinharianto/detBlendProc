from json import dump
import cv2
from cv2 import data
import streamlit as st

import albumentations as A

from control import param2func


def show_credentials():
    st.markdown("* * *")
    st.subheader("Credentials:")
    st.markdown(
        (
            "Source: [github.com/IliaLarchenko/albumentations-demo]"
            "(https://github.com/IliaLarchenko/albumentations-demo)"
        )
    )
    st.markdown(
        (
            "Albumentations library: [github.com/albumentations-team/albumentations]"
            "(https://github.com/albumentations-team/albumentations)"
        )
    )
    st.markdown(
        (
            "Image Source: [pexels.com/royalty-free-images]"
            "(https://pexels.com/royalty-free-images/)"
        )
    )


def show_docstring(obj_with_ds):
    st.markdown("* * *")
    st.subheader("Docstring for " + obj_with_ds.__class__.__name__)
    st.text(obj_with_ds.__doc__)

def show_random_params(data: dict, interface_type: str = "Professional"):
    """Shows random params used for transformation (from A.ReplayCompose)"""
    if interface_type == "Professional":
        st.subheader("Random params used")
        random_values = {}
        for applied_params in data["replay"]["transforms"]:
            random_values[
                applied_params["__class_fullname__"].split(".")[-1]
            ] = applied_params["params"]
        st.write(random_values)
    #changelog
    elif interface_type == "Custom":
        st.subheader("Random params used")
        random_values = {}
        for applied_params in data["replay"]["transforms"]:
            random_values[
                applied_params["__class_fullname__"].split(".")[-1]
            ] = applied_params["params"]
        st.write(random_values)

def get_transformations_params(transform_names: list, augmentations: dict) -> list:
    transforms = []
    for i, transform_name in enumerate(transform_names):
        # select the params values
        st.sidebar.subheader("Params of the " + transform_name)
        param_values = show_transform_control(augmentations[transform_name], i)
        transforms.append(getattr(A, transform_name)(**param_values))
    return transforms

def show_transform_control(transform_params: dict, n_for_hash: int) -> dict:
    #transform_params = augmentation["blur"]
    # [{'defaults': [3, 7], 'limits_list': [3, 100], 'param_name': 'blur_limit', 'type': 'num_interval'}]
    param_values = {"p": 1.0}
    if len(transform_params) == 0:
        st.sidebar.text("Transform has no parameters")
    else:
        for param in transform_params:
            control_function = param2func[param["type"]]
            if isinstance(param["param_name"], list):
                returned_values = control_function(**param, n_for_hash=n_for_hash)
                for name, value in zip(param["param_name"], returned_values):
                    param_values[name] = value
            else:
                param_values[param["param_name"]] = control_function(
                    **param, n_for_hash=n_for_hash
                )
    return param_values

#change log
def get_transformations_params_custom(transform_names: list, augmentations: dict) -> list:

    transforms_all = []
    
    for group, transform_name_group in enumerate(transform_names):
        transforms = []
        st.sidebar.subheader(f"Params of the Group {group}")
        one_param_values = show_transform_control(augmentations["OneOf"], group-100)

        for i, transform_name in enumerate(transform_name_group[1]):
            # select the params values
            st.sidebar.subheader("Params of the " + transform_name)
            param_values = show_transform_control(augmentations[transform_name], i+group*100)
            transforms.append(getattr(A, transform_name)(**param_values))
        transforms_all.append(A.OneOf(transforms, **one_param_values))

    
    return transforms_all

def save_json_data(file_name:str,dict:dict):
    import json
    
    with open(file_name, 'w') as f:
        json.dump(dict,f,indent=4)
    f.close()

def albu_preview(test_transforms):
    
    for item in test_transforms:
        if isinstance(item,list):
            
            st.code(f"{item[0]}")
            temp_oneof =[]
            for aug_names in item[1]:
                st.code(f"      {aug_names}")
                temp_oneof.append(aug_names)      

        else:
            st.code(f"{item}")