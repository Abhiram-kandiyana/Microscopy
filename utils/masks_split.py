import os
import importlib
import shutil

# Import constants module
loader = importlib.machinery.SourceFileLoader( 'mc_constants', r'C:\Users\KAVYA\Abhiram\microscopy\constants\constants.py' )
spec = importlib.util.spec_from_loader( 'mc_constants', loader )
mc_constants = importlib.util.module_from_spec( spec )
loader.exec_module( mc_constants)

train_path= mc_constants.slide1_64X64_train
test_path = mc_constants.slide1_64X64_test
valid_path = mc_constants.slide1_64X64_val
mask_path = mc_constants.slide1_64X64_annotated

mask_train_path = mc_constants.slide1_64X64_masks_train
mask_test_path = mc_constants.slide1_64X64_masks_test
mask_valid_path = mc_constants.slide1_64X64_masks_val

img_dir = mc_constants.image_dir

if(not os.path.exists(mask_train_path)):
    os.makedirs(mask_train_path)

if(not os.path.exists(mask_test_path)):
    os.makedirs(mask_test_path)

if(not os.path.exists(mask_valid_path)):
    os.makedirs(mask_valid_path)

train_img_dir = os.listdir(train_path)
test_img_dir = os.listdir(test_path)
val_img_dir = os.listdir(valid_path)

for i in train_img_dir:
    shutil.copytree(os.path.join(mask_path,img_dir,i),os.path.join(mask_train_path,i))

for i in test_img_dir:
    shutil.copytree(os.path.join(mask_path,img_dir,i),os.path.join(mask_test_path,i))

for i in val_img_dir:
    shutil.copytree(os.path.join(mask_path,img_dir,i),os.path.join(mask_valid_path,i))





