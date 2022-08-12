15 Feb 2021

Changes made
15 Feb 2021 - Added sorting of slices based on slice names in format: 'Z-020.07.bmp' Slices are sorted based on the number between Z and first dot (here, -020). This number shows z value on the z axis and can be negative or positive
Update mode is also implemented. Old masks are preserved and new cells can be added in that. 

python version: 3.5
Main file: VideoDisectorTool
Example directory structure is PI3-20 in this directory
case dir is mouse dir (PI3-20 for example)
Optional: Ref dir is directory with click annotation on EDF images (corresponding to the stack) for reference (in the example: PI3-20/annotation)
Up and down arrow keys: navigate within a stacks
Enter: To go to next stack
Masks are saved in case_annotated directory with same directory structure as case directory
Annotation is stored in ManualAnnotation.json file under case_annotated directory
The tool can be stopped any time. Next time, it starts from where you left.

ManualAnnotation.json structure

Section names -> keys
Values ->[]  list of stack dictionaries (One dictionary per stack)

example:
{"Section1": [{stack1 dictionary},{stack2 dictionary}]}

Stack dictionary structure

{	
	"StackName": name of the stack
	"height": height of images
	"Width": width of images
	"cells": [] -> list of cells (each cell is a dictionary) in this stack
}


Cell dictionary structure

{
	"sliceNo": Slice number in the stack where the cell is best in focus and annotated,
	"centroid": centroid of the drawn cell contour,
	"area": area of the cell contour,
	"contour": [] ->list of [x,y] locations of the drawn contour

} 


18 March 2021
Mask_Annotation_Tool_v3.0 - changed sorting of the slice names. Now based on +/-. Was not needed for NeuN dual stain dataset. Needed for ChAT and NeuN SS dataset.