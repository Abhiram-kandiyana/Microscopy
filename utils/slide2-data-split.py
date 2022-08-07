import os
import random
import shutil
import importlib.machinery
import importlib.util
import json

# Import constants module
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'C:\Users\KAVYA\Abhiram\microscopy\constants\constants.py' )
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)

input_path = os.path.join(mc_constants.slide2_split,mc_constants.image_dir)
annotated_path = os.path.join(mc_constants.slide2_annotated,mc_constants.image_dir) 
visited_path = os.path.join(mc_constants.slide2_annotated,mc_constants.visited_json)

try:
    with open(visited_path, 'r') as read_fp:
        visited = json.load(read_fp)
except:
    print("visited.json is not found")

visited_stacks = visited["stacks"]

visited_stacks_list = []
for visited_stack in visited_stacks:
    visited_stacks_list.append(visited_stack["stackName"])


train_dir_path = os.path.join(input_path,mc_constants.train)
test_dir_path = os.path.join(input_path,mc_constants.test)
val_dir_path = os.path.join(input_path,mc_constants.valid)

if (not os.path.exists(train_dir_path)):
    os.makedirs(train_dir_path)

if (not os.path.exists(test_dir_path)):
    os.makedirs(test_dir_path)

if (not os.path.exists(val_dir_path)):
    os.makedirs(val_dir_path)

annotated_img_dir = os.listdir(annotated_path)

random.seed(44)
random.shuffle(annotated_img_dir)

for stackNo,stackName in enumerate(annotated_img_dir):
    annotated_stack_name = mc_constants.Slide2 + "_" + mc_constants.image_dir + "_" + stackName
    if annotated_stack_name in visited_stacks_list:
        shutil.move(os.path.join(input_path,stackName),test_dir_path)




