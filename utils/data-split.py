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

count_annotation_path = os.path.join(mc_constants.slide2_split,mc_constants.count_annotation,mc_constants.count_annotation_json)
old_count_annotation_path = os.path.join(mc_constants.slide2_split, mc_constants.old + mc_constants.count_annotation,mc_constants.count_annotation_json)

count_annotation_dir = {}
old_count_annotation_dir = {}
try:
    with open(count_annotation_path,'r') as fp:
        count_annotation_dir = json.load(fp)

except:
    print("error while reading count_annotation file at ",count_annotation_path)

try:
    with open(old_count_annotation_path,'r') as fp:
        old_count_annotation_dir = json.load(fp)

except:
    print("error while reading count_annotation file at ",old_count_annotation_path)

        


print(count_annotation_dir)
print(old_count_annotation_dir)

all_stacks =  set(mc_constants.get_stack_names_from_count_annotation_json(count_annotation_dir))

old_stacks = set(mc_constants.get_stack_names_from_count_annotation_json(old_count_annotation_dir))

new_stacks = list(all_stacks - old_stacks)

annotated_path = os.path.join(mc_constants.slide1_64X64_annotated,mc_constants.image_dir) 
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

for stackNo,stackName in enumerate(new_stacks[0:108]):
    shutil.move(os.path.join(input_path,stackName),train_dir_path)

for stackNo,stackName in enumerate(new_stacks[50:100]):
    shutil.move(os.path.join(input_path,stackName),test_dir_path)

for stackNo,stackName in enumerate(new_stacks[108:]):
    shutil.move(os.path.join(input_path,stackName),val_dir_path)




