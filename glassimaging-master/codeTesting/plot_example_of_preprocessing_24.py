import SimpleITK as sitk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import os

# Code that generates visualizations for each individual preprocessing step

def z_score_norm(img_np, otsu=False):
    initial_img_np = img_np
    if otsu:
        img_np = get_ROI_filter(img_np)
    values_nonzero = img_np[np.nonzero(img_np)]
    mean_nonzero = np.mean(values_nonzero)
    std_nonzero = np.std(values_nonzero)
    img_n = (initial_img_np - mean_nonzero) / std_nonzero
    return img_n

def i_scaling(img_np, otsu=False):
    initial_img_np = img_np
    if otsu:
        img_np = get_ROI_filter(img_np)
    values_nonzero = img_np[np.nonzero(img_np)]
    LIR = np.percentile(values_nonzero.flatten(), 50)
    HIR = np.percentile(values_nonzero.flatten(), 90)
    img_n = (initial_img_np - LIR) / (HIR - LIR)
    return img_n


def resample_img2(image, reference_img, resampling_factor, use_seg=False):

    img_spacing = reference_img.GetSpacing()
    img_direction = reference_img.GetDirection()
    img_origin = reference_img.GetOrigin()
    img_size = reference_img.GetSize()
    img_pixelIDValue = reference_img.GetPixelIDValue()

    new_img_size = tuple(int(i / resampling_factor) for i in img_size)
    new_img_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_img_size, img_size, img_spacing)]

    resample_to_this_image = sitk.Image(*new_img_size, img_pixelIDValue)

    resample_to_this_image.SetSpacing(new_img_spacing)

    image_to_resample = sitk.GetImageFromArray(image)

    if not use_seg:
        resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkBSplineResamplerOrder3)
    else:
        resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkNearestNeighbor)

    resampled_img_np = sitk.GetArrayFromImage(resampled_img)

    return resampled_img_np

def get_ROI_filter(img):
    if not isinstance(img, sitk.Image):
        # convert img np array back to sitk image
        img = sitk.GetImageFromArray(img)

    # Get the ROI by using otsu based thresholding approach
    # first use straightforward otsu (this doesnt yield perfect results as threshold is too high)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    ROI = otsu_filter.Execute(img)

    # locate the negated parts of the image
    otsu_filter2 = sitk.OtsuThresholdImageFilter()
    otsu_filter2.SetInsideValue(1)
    otsu_filter2.SetOutsideValue(0)
    ROI_neg = otsu_filter2.Execute(img)

    mask_filter = sitk.MaskImageFilter()
    masked_ROI_neg = mask_filter.Execute(img, ROI_neg)

    # use otsu again on the negated part to find values that are close to empty space but are not empty space
    otsu_filter3 = sitk.OtsuThresholdImageFilter()
    otsu_filter3.SetInsideValue(0)
    otsu_filter3.SetOutsideValue(1)
    ROI_addition = otsu_filter3.Execute(masked_ROI_neg)

    # Add two passes of otsu together here
    combine_filter = sitk.AddImageFilter()
    combined_filters = combine_filter.Execute(ROI, ROI_addition)

    combined_filters_np = sitk.GetArrayFromImage(combined_filters)

    original_img_np = sitk.GetArrayFromImage(img)

    # use a numpy mask for excluding irrelevant data
    mx = np.ma.masked_array(original_img_np, mask=np.logical_not(combined_filters_np))
    return mx

def plot_image_of(img_np, combi, image_location, save_path):
    norm_method = combi[0]
    otsu = combi[1]
    resampling_factor = combi[2]

    if norm_method == 'z-score':
        img_np = z_score_norm(img_np, otsu)
    elif norm_method == 'i-scaling':
        img_np = i_scaling(img_np, otsu)

    if not resampling_factor == 1:
        img_np = resample_img2(img_np, sitk.ReadImage(image_location), resampling_factor, use_seg=False)


    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    middle_slice_of_image = img_np.shape[0] // 2
    axs[0].imshow(img_np[middle_slice_of_image, :, :], cmap='gray')
    axs[1].hist(img_np.ravel(), 75)
    plt.savefig(save_path, format='png')
    plt.close()

def plot_example_of_preprocessing_24(image_location):
    img = sitk.ReadImage(image_location)
    img_np = sitk.GetArrayFromImage(img)

    normalization_methods = ['i-scaling']
    otsu = [False]
    resampling_factors = [1]
    n4bc = ['noN4BC']

    all_combinations = list(itertools.product(normalization_methods, otsu, resampling_factors, n4bc))

    print(all_combinations)
    for combi in all_combinations:
        file_name = str(combi[0]) + '_' + str(combi[1]) + '_' + str(combi[2]) + '_' + str(combi[3]) + 'nieuw2.png'
        print('working on:', combi, 'with name:', file_name)
        file_loc_and_name = os.path.join('C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\codeTesting\\example_imgs_of_preprocessing_24', file_name)
        plot_image_of(img_np, combi, image_location, save_path=file_loc_and_name)

def plot_example_of_preprocessing_1(image_location):
    normalization_methods = ['z-score', 'i-scaling']
    otsu = [True, False]
    resampling_factors = [1,2,4]
    n4bc = ['noN4BC']

    all_combinations = list(itertools.product(normalization_methods, otsu, resampling_factors, n4bc))

    print(all_combinations)
    index = 0
    fig, axs = plt.subplots(5, 2, figsize=(10, 16))
    for combi in all_combinations:
        img = sitk.ReadImage(image_location)
        img_np = sitk.GetArrayFromImage(img)

        if combi == ('z-score', False, 1, 'noN4BC') or combi == ('z-score', True, 1, 'noN4BC') or combi == ('i-scaling', False, 1, 'noN4BC') or combi == ('z-score', False, 1, 'withN4BC') or combi == ('z-score', False, 4, 'noN4BC'):
            file_name = 'combined.png'
            print('working on:', combi, 'with name:', file_name)
            file_loc_and_name = os.path.join('C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\codeTesting\\example_imgs_of_preprocessing_24', file_name)

            norm_method = combi[0]
            otsu = combi[1]
            resampling_factor = combi[2]

            if norm_method == 'z-score':
                img_np = z_score_norm(img_np, otsu)
            elif norm_method == 'i-scaling':
                img_np = i_scaling(img_np, otsu)

            if not resampling_factor == 1:
                img_np = resample_img2(img_np, sitk.ReadImage(image_location), resampling_factor, use_seg=False)

            middle_slice_of_image = img_np.shape[0] // 2
            axs[index, 0].imshow(img_np[middle_slice_of_image, :, :], cmap='gray')
            axs[index, 1].hist(img_np.ravel(), 75)

            index = index + 1

    plt.savefig(file_loc_and_name, format='png')
    plt.close()

if __name__ == '__main__':
    image_location = 'C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\img_sample_for_vis\\2310_BTD-0002.nii.gz'
    plot_example_of_preprocessing_24(image_location)

