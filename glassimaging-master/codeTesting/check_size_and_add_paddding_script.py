import SimpleITK as sitk
import numpy as np
import os

img = sitk.ReadImage("C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\img_sample_for_vis\\2310_BTD-0002.nii.gz")
# get array form image
img_np = sitk.GetArrayFromImage(img)
img_np = np.expand_dims(img_np, axis=0)

# Get original image size
img_size = img_np.shape

print(img_size)
resampling_factor = 1
# check if img is larger than default patch size for resampling factor 1
if resampling_factor == 1:
    new_img_size = tuple([i if i > 112 or i == 1 else 112 for i in img_size[-3:]])
    new_img_size = img_size[:-3] + new_img_size
elif resampling_factor == 2:
    new_img_size = tuple([i if i > 56 or i == 1 else 56 for i in img_size[-3:]])
    new_img_size = img_size[:-3] + new_img_size
elif resampling_factor == 4:
    new_img_size = tuple([i if i > 28 or i == 1 else 28 for i in img_size[-3:]])
    new_img_size = img_size[:-3] + new_img_size

# Here we first create a new image of zeroes and then fill it with the sitk image values
padded_img_np = np.zeros(new_img_size)
padded_img_np[:img_size[0], :img_size[1], :img_size[2], :img_size[3]] = img_np

print(new_img_size)



def set_new_patch_size(min_img_size, data_loc):
    res_factor = int(data_loc.split(os.sep)[-1].split("Res")[-1])

    min_img_size = [i if i >= 112/res_factor else int(112/res_factor) for i in min_img_size]

    patch_size = [(i//16)*16 if i < 112 else 112 for i in min_img_size]
    # print(min_img_size)
    print(patch_size)

set_new_patch_size((512,512,74), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes1")
set_new_patch_size((256,256,37), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes2")
set_new_patch_size((128,128,18), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes4")

set_new_patch_size((440,440,74), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes1")
set_new_patch_size((220,220,37), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes2")
set_new_patch_size((110,110,18), "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTSRes4")
