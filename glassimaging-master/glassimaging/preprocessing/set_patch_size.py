import os
import SimpleITK as sitk
import sys
import json

# Code that is used to set the patch size based on the resampled image size. It now works for the BTD, LiTS and Ergo dataset. Pay attention to whether you are using resampling on the fly or if this is already done for the whole dataset (see line 50).
def set_patch_size(config):
    data_loc = config["Nifti Source"]
    lstFilesGz = []  # create an empty list with images
    for dirName, subdirList, fileList in os.walk(data_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                # check whether a file has a segmentation from a specific person
                lstFilesGz.append(os.path.join(dirName, filename))

    # get the specified resampling factor, an image from the dataset, its size and its dimension
    resampling_factor = config["resampling_factor"]

    for index, img_path in enumerate(lstFilesGz):
        # read image
        sample_img = sitk.ReadImage(img_path)
        # when in the first loop set the min_size to the first img size
        if index == 0:
            min_size = sample_img.GetSize()
            sample_img_dim = sample_img.GetDimension()
        # get the current img_size
        sample_img_size = sample_img.GetSize()
        # get a tuple of the smallest sizes.
        min_size = tuple([x if x <= y else y for x, y in zip(min_size, sample_img_size)])

    # create a new patch size variable that will be returned
    patch_size = [0] * sample_img_dim

    # get the size of the image after the resampling factor has been applied
    resampled_img_size = tuple([int(i/resampling_factor) for i in min_size])

    # create a list which indicates what items from the config patch size are actually too large to use
    change_list = list(i > j for i, j in zip(config["Patch size"], resampled_img_size))

    # given the change list we can now set the patch sizes to the right values
    for index, i in enumerate(change_list):
        if i:
            # get closest multiple of 16 as this is needed for ensuring that all layers of the network can be run
            cmo16 = resampled_img_size[index]//16
            # multiply cmo16 with 16 to create a size that is the closest multiple of 16
            patch_size[index] = cmo16*16
        else:
            # if no change is needed just use the original value
            patch_size[index] = config["Patch size"][index]
    # in the case we're using the Lits/Ergo data we need to be aware that resampling factor shouldn't be taken into account
    # as it is always one due tot he fact that the resampled data is provided instead of doing resampling on the fly
    # as is the case with BTD
    if config["Dataset"] == "LitsData":
        res_factor = int(data_loc.split(os.sep)[-1].split("Res")[-1])
        # min_img_size = [i if i >= 112 / res_factor else int(112 / res_factor) for i in min_size]
        min_img_size = [i if i >= 112 else 112 for i in min_size]
        patch_size = [(i // 16) * 16 if i < config['Patch size'][index] else config['Patch size'][index] for index, i in enumerate(min_img_size)]
    elif config["Dataset"] == "ErgoData":
        min_img_size = [i if i >= 112 else 112 for i in min_size]
        patch_size = [(i // 16) * 16 if i < config['Patch size'][index] else config['Patch size'][index] for index, i in
                      enumerate(min_img_size)]
    if any(change_list):
        print('patch size changed from:', config['Patch size'], 'to:', patch_size, 'due to image being too small for set patch size')

    return patch_size


# This piece of code can be used to test the function above in order to make sure that it works as expected.
# In future maybe build test cases to do this
# if __name__ == '__main__':
#     set_patch_size({"Nifti Source": "C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\img_sample_for_vis\\sub",
#                     "Patch size": [
#                         112,
#                         112,
#                         112
#                     ]})
