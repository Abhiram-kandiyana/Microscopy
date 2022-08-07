# function to sort list of slice names in a stack. This function to be used as key in inbuilt function sorted() it
# assumes slice names in formate: 'Z-020.07.bmp' Slices are sorted based on the number between Z and first dot (here,
# -020). This number shows z value on the z axis and can be negative or positive

def sort_slice_name_lst(name):
    dot_idx = name.find('.')
    return int(name[1:dot_idx])