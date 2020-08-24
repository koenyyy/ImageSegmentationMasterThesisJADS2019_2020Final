import os
import SimpleITK as sitk
import numpy as np


def res_using_spacing_factors(image, res_factor, filename):
    old_img_pixelIDValue = image.GetPixelIDValue()
    dimension = image.GetDimension()
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    old_img_spacing = image.GetSpacing()
    new_img_spacing = [i * res_factor for i in old_img_spacing]

    old_img_size = image.GetSize()
    new_img_size = tuple(int(old_spacing * old_size / new_spacing) for old_spacing, old_size, new_spacing in
                         zip(old_img_spacing, old_img_size, new_img_spacing))

    resample_to_this_image = sitk.Image(*new_img_size, old_img_pixelIDValue)

    resample_to_this_image.SetSpacing(new_img_spacing)
    resample_to_this_image.SetOrigin(reference_origin)
    resample_to_this_image.SetDirection(reference_direction)
    reference_center = np.array(
        resample_to_this_image.TransformContinuousIndexToPhysicalPoint(np.array(resample_to_this_image.GetSize()) / 2.0))

    #################

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(reference_direction)
    transform.SetTranslation(np.array(reference_origin) - reference_origin)

    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(old_img_size) / 2.0))
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    #################
	# Here we use dataset specific names to find the segmentations and the masks such that we can perform resampling. we make use of 3rd order spline interpolation for the resampling of images and neares neighbor for segmentations
    if "T1_T2_Morphology" in filename:
        resampled_img = sitk.Resample(image, resample_to_this_image, centered_transform,
                                      sitk.sitkBSplineResamplerOrder3, 0.0)
    elif "_bone_mask" in filename:
        resampled_img = sitk.Resample(image, resample_to_this_image, centered_transform,
                                      sitk.sitkNearestNeighbor, 0.0)

    # print(old_img_spacing, old_img_size)
    # print(new_img_spacing, new_img_size)


    return resampled_img


def save_nii(image, curr_nii_loc, res_factor_used):
    filename = curr_nii_loc.split(os.sep)[-1]
    save_folder = os.path.join(*curr_nii_loc.replace("Ergo_Data_normal", "Ergo_data_Res" + str(res_factor_used)).split(os.sep)[:-1])

    # save_folder = os.path.join(*curr_nii_loc.split(os.sep)[:-2], "Ergo_Res" + str(res_factor_used))
    save_file = os.path.join(save_folder, filename)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    sitk.WriteImage(image, save_file)


def resample_dataset(data_loc):
    nii_list = []
    for dirName, subdirList, fileList in os.walk(data_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                nii_list.append(os.path.join(dirName, filename))

    for nii_loc in nii_list:
        print('resampling:', nii_loc)
        image = sitk.ReadImage(nii_loc)
        filename = nii_loc.split(os.sep)[-1]
		# here we call the resampling functions using an input image, spacing factor and filename based on current file name
        res_image_1 = res_using_spacing_factors(image, 1, filename)
        res_image_2 = res_using_spacing_factors(res_image_1, 2, filename)
        res_image_4 = res_using_spacing_factors(res_image_1, 4, filename)

        save_nii(res_image_1, nii_loc, 1)
        save_nii(res_image_2, nii_loc, 2)
        save_nii(res_image_4, nii_loc, 4)


if __name__ == '__main__':
# make sure to adjust the names of the location
    data_location = "/media/data/kderaad/Ergo_Data_normal"
    resample_dataset(data_location)
    print("done resampling")

