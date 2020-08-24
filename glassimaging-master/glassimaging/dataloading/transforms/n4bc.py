import SimpleITK as sitk
import numpy as np


class N4BiasCorrection(object):
    """Use bias correction to improve images subjected to bias field signals (low-frequency and smooth signals).


    """

    # bias correction is not used on the fly as it takes too long. Prepare your dataset using bias correction in advance using for example the cpu cluster.
    def __call__(self, sample):
        image = sample['data']
        image = sitk.GetImageFromArray(image)
        print('started N4')

        initial_img = image
        img_size = initial_img.GetSize()
        img_spacing = initial_img.GetSpacing()

        image = sitk.Cast(image, sitk.sitkFloat64)

        image = sitk.GetArrayFromImage(image)
        image[image == 0] = np.finfo(float).eps
        image = sitk.GetImageFromArray(image)

        # Not needed here (is needed in glass imagaing project)
        # reset the origin and direction to what it was initially
        image.SetOrigin(initial_img.GetOrigin())
        image.SetDirection(initial_img.GetDirection())
        image.SetSpacing(initial_img.GetSpacing())

        print('working on N4')
        maskImage = sitk.OtsuThreshold(image, 0, 1)

        shrink_factor = [(img_size[0] // 128 if img_size[0] % 128 is not img_size[0] else 1),
                         (img_size[1] // 128 if img_size[1] % 128 is not img_size[1] else 1),
                         (img_size[2] // 128 if img_size[2] % 128 is not img_size[2] else 1)]


        shrink_filter = sitk.ShrinkImageFilter()
        image_shr = shrink_filter.Execute(image, shrink_factor)
        maskImage_shr = shrink_filter.Execute(maskImage, shrink_factor)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image_shr = corrector.Execute(image_shr, maskImage_shr)

        # extract the bias field
        exp_logBiasField = image_shr / corrected_image_shr

        # resample the bias field to match original image
        reference_image2 = sitk.Image(img_size, exp_logBiasField.GetPixelIDValue())
        reference_image2.SetOrigin(initial_img.GetOrigin())
        reference_image2.SetDirection(initial_img.GetDirection())
        reference_image2.SetSpacing(img_spacing)

        resampled_exp_logBiasField = sitk.Resample(exp_logBiasField, reference_image2)

        # print('Checkpoint img', image.GetPixelIDTypeAsString(), resampled_exp_logBiasField.GetPixelIDTypeAsString())
        # print('Checkpoint img', image.GetSize(), image.GetOrigin(), image.GetDirection(),
        #       resampled_exp_logBiasField.GetSize(), resampled_exp_logBiasField.GetOrigin(), resampled_exp_logBiasField.GetDirection())

        divide_filter2 = sitk.DivideImageFilter()
        corrected_image = divide_filter2.Execute(image, resampled_exp_logBiasField)

        # used for debugging
        # image_np = sitk.GetArrayFromImage(image)
        # corrected_image_np = sitk.GetArrayFromImage(corrected_image)
        # resampled_exp_logBiasField_np = sitk.GetArrayFromImage(resampled_exp_logBiasField)
        # image_shr_np = sitk.GetArrayFromImage(image_shr)
        # corrected_image_shr_np = sitk.GetArrayFromImage(corrected_image_shr)
        # exp_logBiasField_np = sitk.GetArrayFromImage(exp_logBiasField)

        # fig, axs = plt.subplots(3, 2)
        # axs[0, 0].imshow(image_np[35])
        # axs[1, 0].imshow(corrected_image_np[35])
        # axs[2, 0].imshow(resampled_exp_logBiasField_np[35])
        #
        # axs[0, 1].imshow(image_shr_np[35])
        # axs[1, 1].imshow(corrected_image_shr_np[35])
        # axs[2, 1].imshow(exp_logBiasField_np[35])
        # plt.plot()
        # plt.show()
        #
        # print(image_shr_np[35, 42, 47], corrected_image_shr_np[35, 42, 47], exp_logBiasField_np[35, 42, 47])
        # print(image_shr_np[35, 42, 45], corrected_image_shr_np[35, 42, 45], exp_logBiasField_np[35, 42, 45])

        return {'name_img': name_img, 'name_seg': name_seg, 'image': corrected_image, 'segmentation': segmentation}
