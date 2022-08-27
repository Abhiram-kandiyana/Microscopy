import importlib
import random
# Import constants module
import os
import shutil

random.seed(44)

loader = importlib.machinery.SourceFileLoader('mc_constants',r'/Users/abhiramkandiyana/Microscopy/constants/constants.py')
spec = importlib.util.spec_from_loader('mc_constants', loader)
mc_constants = importlib.util.module_from_spec(spec)
loader.exec_module(mc_constants)

loader = importlib.machinery.SourceFileLoader( 'common_util', r'/Users/abhiramkandiyana/Microscopy/utils/common_utils.py' )
spec = importlib.util.spec_from_loader('common_util', loader )
common_util = importlib.util.module_from_spec( spec )
loader.exec_module( common_util)


slide1_name = mc_constants.Slide1
slide2_name = mc_constants.Slide2
img_dir = mc_constants.image_dir


slide1_stacks = os.listdir(os.path.join(mc_constants.inputImages,slide1_name,img_dir))
slide2_stacks = os.listdir(os.path.join(mc_constants.inputImages,slide2_name,img_dir))

slide1_count_annotated_stacks = common_util.get_stack_names_from_count_annotation_json(os.path.join(mc_constants.inputImages,mc_constants.testing_dir,mc_constants.mask_annotation_automation_dir,slide1_name))
slide2_count_annotated_stacks = common_util.get_stack_names_from_count_annotation_json(os.path.join(mc_constants.inputImages,mc_constants.testing_dir,mc_constants.mask_annotation_automation_dir,slide2_name))



slide1_len  = len(slide1_count_annotated_stacks)
slide2_len = len(slide2_count_annotated_stacks)

new_img_dir_path = os.path.join(mc_constants.inputImages,mc_constants.testing_dir,mc_constants.mask_annotation_automation_dir,slide2_name,img_dir)
new_img_dir_path_1 = os.path.join(mc_constants.inputImages,mc_constants.testing_dir,mc_constants.mask_annotation_automation_dir,slide1_name,img_dir)


# if not os.path.exists(new_img_dir_path):
#     os.makedirs(new_img_dir_path)
#
# if not os.path.exists(new_img_dir_path_1):
#     os.makedirs(new_img_dir_path_1)

for i in range(12):
    j=random.randint(0,slide2_len-1)
    k=random.randint(0,slide1_len-1)
    img_path = os.path.join(mc_constants.inputImages,slide2_name,img_dir,slide2_count_annotated_stacks[j])
    img_path1 = os.path.join(mc_constants.inputImages, slide1_name, img_dir, slide1_count_annotated_stacks[k])
    new_img_path  = os.path.join(new_img_dir_path,slide2_count_annotated_stacks[j])
    new_img_path1 = os.path.join(new_img_dir_path_1, slide1_count_annotated_stacks[k])
    shutil.copytree(img_path,new_img_path)
    shutil.copytree(img_path1,new_img_path1)
