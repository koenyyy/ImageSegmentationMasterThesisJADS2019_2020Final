import os
import SimpleITK as sitk
import numpy as np

# bias correction is not used on the fly as it takes too long. Prepare your dataset using bias correction in advance using this code on your dataset or make use of the CPU cluster which is much faster and more accurate as here we work with more estimates.
def apply_bias_correction(img:sitk.Image):
    # print('working on N4')
    initial_img = img
    img_size = initial_img.GetSize()
    img_spacing = initial_img.GetSpacing()
    img_pixel_ID = img.GetPixelID()

    # Cast to float to enable bias correction to be used
    image = sitk.Cast(img, sitk.sitkFloat64)

    image = sitk.GetArrayFromImage(image)
    image[image == 0] = np.finfo(float).eps
    image = sitk.GetImageFromArray(image)

    # reset the origin and direction to what it was initially
    image.SetOrigin(initial_img.GetOrigin())
    image.SetDirection(initial_img.GetDirection())
    image.SetSpacing(initial_img.GetSpacing())

    maskImage = sitk.OtsuThreshold(image, 0, 1)

    # Calculating a shrink factor that will be used to reduce image size and increase N4BC speed
    shrink_factor = [(img_size[0] // 64 if img_size[0] % 128 is not img_size[0] else 1),
                     (img_size[1] // 64 if img_size[1] % 128 is not img_size[1] else 1),
                     (img_size[2] // 64 if img_size[2] % 128 is not img_size[2] else 1)]

    # shrink the image and the otsu masked filter
    shrink_filter = sitk.ShrinkImageFilter()
    image_shr = shrink_filter.Execute(image, shrink_factor)
    maskImage_shr = shrink_filter.Execute(maskImage, shrink_factor)

    # apply image bias correction using N4 bias correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image_shr = corrector.Execute(image_shr, maskImage_shr)

    # extract the bias field by dividing the shrunk image by the corrected shrunk image
    exp_logBiasField = image_shr / corrected_image_shr

    # resample the bias field to match original image
    reference_image2 = sitk.Image(img_size, exp_logBiasField.GetPixelIDValue())
    reference_image2.SetOrigin(initial_img.GetOrigin())
    reference_image2.SetDirection(initial_img.GetDirection())
    reference_image2.SetSpacing(img_spacing)
    resampled_exp_logBiasField = sitk.Resample(exp_logBiasField, reference_image2)

    # extract the corrected image by dividing the initial image by the resampled bias field that was calculated earlier
    divide_filter2 = sitk.DivideImageFilter()
    corrected_image = divide_filter2.Execute(image, resampled_exp_logBiasField)

    # cast back to initial type to allow for further processing
    corrected_image = sitk.Cast(corrected_image, img_pixel_ID)

    return corrected_image


# bias correction is not used on the fly as it takes too long. Prepare your dataset using bias correction in advance using this code on your dataset or make use of the CPU cluster which is much faster and more accurate as here we work with more estimates.
def correct_dataset():
    dataset_loc = 'C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Data\\BTD'
    save_loc = 'C:\\Users\\s145576\\Documents\\GitHub\\master_thesis\\glassimaging-master\\BTD_N4BC'

    # Listing all the names of the images and segmentations
    lstFilesGz = []  # create an empty list with images
    lstFilesGzSeg = []

    for dirName, subdirList, fileList in os.walk(dataset_loc):
        for filename in fileList:
            if "nii.gz" in filename.lower():  # check whether the file's .gz
                # check whether a file has a segmentation from a specific person
                if not '_mask' in filename.lower():
                    lstFilesGz.append(os.path.join(dirName, filename))
                if '_mask' in filename.lower():
                    lstFilesGzSeg.append(os.path.join(dirName, filename))


    for file in lstFilesGz:
        # get the folder structure for which the dataloader is built
        respective_folder_structure = file.split(str(os.sep) + 'BTD' + str(os.sep))[-1].split(os.sep)[0]
        # get the filename
        file_name = file.split(os.sep)[-1]

        # read the image that needs bias correction
        img = sitk.ReadImage(file)

        img_bc = apply_bias_correction(img)

        # specify the folder structure which is needed for storing the corrected image
        save_folder = os.path.join(save_loc, respective_folder_structure)
        # specify the folder structure and add the filename which are needed for storing the corrected image
        save_file = os.path.join(save_folder, file_name)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        sitk.WriteImage(img_bc, save_file)

    for file in lstFilesGzSeg:
        # get the folder structure for which the dataloader is built
        respective_folder_structure = file.split(str(os.sep) + 'BTD' + str(os.sep))[-1].split(os.sep)[0]
        # get the filename
        file_name = file.split(os.sep)[-1]

        # read the image that needs bias correction
        img = sitk.ReadImage(file)

        # specify the folder structure which is needed for storing the corrected image
        save_folder = os.path.join(save_loc, respective_folder_structure)
        # specify the folder structure and add the filename which are needed for storing the corrected image
        save_file = os.path.join(save_folder, file_name)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        sitk.WriteImage(img, save_file)

if __name__ == '__main__':
    correct_dataset()
