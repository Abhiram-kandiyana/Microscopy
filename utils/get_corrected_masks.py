import os
import random
import shutil
import importlib.machinery
import importlib.util
import json
import numpy as np
from common_utils import get_stack_names_from_visited_json


# Import constants module
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'C:\Users\KAVYA\Abhiram\microscopy\constants\constants.py' )
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)
image_dir = mc_constants.image_dir

slide = mc_constants.Slide2

main_dir = mc_constants.slide2_annotated

masks_dir = os.path.join(main_dir,mc_constants.image_dir)

masks_list = os.listdir(masks_dir)

visited_dir_path = os.path.join(main_dir,mc_constants.visited_stacks_dir)

visited_masks_list = os.listdir(visited_dir_path)

if not os.path.exists(visited_dir_path):
    os.makedirs(visited_dir_path)

visited_json_path = os.path.join(main_dir,mc_constants.visited_json)

visited_stacks_list = get_stack_names_from_visited_json(visited_json_path)

for i in masks_list:
    stack_name_full = slide+'_'+image_dir+'_'+i
    if stack_name_full in visited_stacks_list and i not in visited_masks_list:
        shutil.copytree(os.path.join(masks_dir,i),os.path.join(visited_dir_path,i))













