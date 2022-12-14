import os
import random
import shutil
import importlib.machinery
import importlib.util
import json
import numpy as np



# Import constants module
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'/Users/abhiramkandiyana/Microscopy/constants/constants.py' )
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)

def get_stack_names_from_count_annotation_json(dir_path):
    path = os.path.join(dir_path,mc_constants.count_annotation,mc_constants.count_annotation_json)

    try:
        with open(path,'r') as fp:
            dict = json.load(fp)
    except Exception as e:
        print("error while reading the count annotation json {}".format(e))
        return

    dict = dict[mc_constants.image_dir]
    stackNames = []
    for stackNo,stack in enumerate(dict):
        stackName_split = stack[mc_constants.stack_name_const].split('_')
        stackNames.append(stackName_split[-2]+'_'+stackName_split[-1])
    return stackNames

def get_stack_names_from_visited_json(path):
    try:
        with open(path, 'r+') as fp:
            visited_json = json.load(fp)

    except:
        print("The {} file does not exist ",path)

    visited_stacks = np.array(visited_json[mc_constants.stacks_const])
    visited_stack_names = [ stack[mc_constants.small_stack_name_const] for stack in visited_stacks ]
    return visited_stack_names
def check_value(value,limit):
    if(value < 0):
        value = 0
    elif(value > limit):
        value = limit
    return value