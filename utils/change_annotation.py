import json


path = './ManualAnnotation.json'

try:
    with open(path, 'r') as ref_fp:
        annotation_dict = json.load(ref_fp)
except:
    print("Cannot read the json file")

print(annotation_dict["NeoCx"])

for stack in annotation_dict['NeoCx']:
    for cell in stack['cells']:
        cell['sliceNo']=9-cell['sliceNo']

with open ("./ManualAnnotation.json","w") as annotation_file:

    json.dump(annotation_dict,annotation_file)
