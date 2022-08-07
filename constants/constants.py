
slide1_64X64_split = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64-split"
slide2_split =  r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-split"
slide1_64X64_annotated = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64_1_annotated"
slide2_annotated = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2_annotated"
slide1_64X64_train = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64-split\NeoCx\train"
slide2_train = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-split\NeoCx\train"
slide1_64X64_test = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64-split\NeoCx\test"
slide2_test = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-split\NeoCx\test"
slide1_64X64_val = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\slide1-64x64-split\NeoCx\val"
slide2_val = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-split\NeoCx\val"

count_annotation = "count_annotaion"

count_annotation_json = "ManualAnnotation.json"

visited_json = "visited.json"

slide1_64X64_masks_train  =r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\masks-split\NeoCx\train"

testing_img_dir = r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\testing\NeoCx"

testing_masks_dir = r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\testing\masks"

slide1_64X64_masks_test  = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\masks-split\NeoCx\test"

slide2_masks_test  = r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\Slide2-masks-split\NeoCx\test"

slide1_64X64_masks_val  =r"C:\Users\KAVYA\Abhiram\microscopy\16bitimages\masks-split\NeoCx\val"

marked_images_path = r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\marked_images_latest"

csv_results_path = r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\csv_results"

predicted_clusters_images_path = r"C:\Users\KAVYA\Abhiram\microscopy\mean-shift\custom_ms_results"



train = "train"
test = "test"
valid = "val"
algo_test = "testing"
image_dir = "NeoCx"
old = "old_"
Slide1_64x64_1 ="slide1-64x64_1"
Slide2 = "Slide2"
stack_name_const = "StackName"
small_stack_name_const = "stackName"
stacks_const = "stacks"
stack_length = 10
visited_stacks_dir = "visited_stacks"

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
