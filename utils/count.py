

import json

count_ann_path = "/Users/abhiram/microscopy/16bitimages/slide1-64x64_1/count_annotaion/ManualAnnotation.json"

mask_ann_path = "/Users/abhiram/microscopy/16bitimages/slide1-64x64_1_annotated/visited.json"

try:
    with open(count_ann_path, 'r') as ref_fp:
        count_ann_dict = json.load(ref_fp)
except:
    print('Reference annotation json not available.')


try:
    with open(mask_ann_path, 'r') as ref_fp:
        mask_ann_dict = json.load(ref_fp)
except:
    print('Reference annotation json not available.')


count_ann_dict = count_ann_dict["Neo_cx"]
# print("Mask annotation dict",mask_ann_dict)
mask_ann_dict = mask_ann_dict["stacks"]

print("stacks in visited",len(mask_ann_dict))

count=0
for stack in mask_ann_dict:
    stackName = stack["stackName"]

    for stack1 in count_ann_dict:
        if stackName == stack1["StackName"]:
            count+=stack1["count"]

print("total count",count)








