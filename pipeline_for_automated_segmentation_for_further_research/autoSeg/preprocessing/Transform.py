import torch
from scipy.ndimage import zoom
from autoSeg.data_loading.LipoDataset import LipoDataset
import SimpleITK as sitk
import numpy as np
# import nipype
# from nipype.interfaces.fsl import BET

import matplotlib.pyplot as plt


class OtsuCrop(object):
    """Crop the black parts of the image away to ensure only relevant parts are used

    Args:

    """
    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        image = sample.get('image')
        segmentation = sample.get('segmentation')

        # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
        # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
        inside_value = 0
        outside_value = 1
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)

        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        cropped_image = sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                              bounding_box[0:int(len(bounding_box) / 2)])
        cropped_segmentation = sitk.RegionOfInterest(segmentation, bounding_box[int(len(bounding_box) / 2):],
                                                     bounding_box[0:int(len(bounding_box) / 2)])

        return {'name_img': name_img, 'name_seg': name_seg, 'image': cropped_image, 'segmentation': cropped_segmentation}


class N4BiasCorrection(object):
    """Use bias correction to improve images subjected to bias field signals (low-frequency and smooth signals).


    """

    def __call__(self, sample):
        print('started N4')
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        image = sample.get('image')
        segmentation = sample.get('segmentation')

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

        print('Checkpoint img', image.GetPixelIDTypeAsString(), resampled_exp_logBiasField.GetPixelIDTypeAsString())
        print('Checkpoint img', image.GetSize(), image.GetOrigin(), image.GetDirection(),
              resampled_exp_logBiasField.GetSize(), resampled_exp_logBiasField.GetOrigin(), resampled_exp_logBiasField.GetDirection())

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

# TODO add mask based normalization
class Normalize(object):
    """Normalize the image in a sample.

        Args:
            norm_method (str): Desired method to use for image normalization. Possible methods are:
                "min_max":          Min-Max Normalization
                "z-score":          Z-Score Normalization (x - mean / std)
                "is_norm":          Intensity score normalization (using low intensity range and high intensity range)
                "tanh_norm":        TanH normalization (doesnt take into account the hampel estimator)
                "mean_norm":        Mean normalization (comparable ti min-max normalization only with mean)
                "hist_norm":        Histogram normalization

        """

    def __init__(self, norm_method: str, lir=1, hir=99, masked: bool = False):
        self.norm_method = norm_method
        self.LIR = lir
        self.HIR = hir
        # masked preprocessing is still very slow due to additional operations
        self.masked = masked

    def min_max_norm(self, img_array):
        #instantiate an array on which operations should be done such as min max
        op_img_array = img_array
        if self.masked:
            # get ROI as a masked np array
            op_img_array = getROIFilter().execute(op_img_array)
            flat_op_img_array = op_img_array.flatten()
            a_min = np.ma.min(flat_op_img_array)
            a_max = np.ma.max(flat_op_img_array)
        else:
            a_min = np.min(op_img_array)
            a_max = np.max(op_img_array)
        image = (img_array - a_min) / (a_max - a_min)
        return image

    def z_score_norm(self, img_array):
        # instantiate an array on which operations should be done such as min max
        op_img_array = img_array
        if self.masked:
            # get ROI as a masked np array
            op_img_array = getROIFilter().execute(op_img_array)
            flat_op_img_array = op_img_array.flatten()
            a_mean = np.ma.mean(flat_op_img_array)
            a_std = np.ma.std(flat_op_img_array)
        else:
            a_mean = float(np.mean(op_img_array, axis=None, dtype=np.float64))
            a_std = float(np.std(op_img_array, axis=None, dtype=np.float64))
        image = (img_array - a_mean) / a_std
        return image

    def is_norm(self, img_array, pct_low=1, pct_high=99):
        op_img_array = img_array
        if self.masked:
            # get ROI as a masked np array
            op_img_array = getROIFilter().execute(op_img_array)
            flat_op_img_array = op_img_array.flatten()
            # here we need to compress the flattened masked array to ensure only unmasked values are used for the percentile calculations
            LIR = np.percentile(flat_op_img_array.compressed(), pct_low)
            HIR = np.percentile(flat_op_img_array.compresed(), pct_high)
        else:
            LIR = np.percentile(op_img_array.flatten(), pct_low)
            HIR = np.percentile(op_img_array.flatten(), pct_high)
        print("intensity norm percentiles:", LIR, HIR)
        image = (img_array - LIR) / (HIR - LIR)
        print(np.min(image), np.max(image))
        return image

    # this is a modifed version that doesnt take into account the hampel estimator that the original method does
    def tanH_norm(self, img_array, pct_low=10, pct_high=90):
        op_img_array = img_array
        if self.masked:
            op_img_array = img_array[np.nonzero(img_array)]
            # get ROI as a masked np array
            op_img_array = getROIFilter().execute(op_img_array)
            flat_op_img_array = op_img_array.flatten()
            a_mean = np.ma.mean(flat_op_img_array)
            a_std = np.ma.std(flat_op_img_array)
        else:
            a_mean = np.mean(op_img_array)
            a_std = np.std(op_img_array)
        image = 0.5 * (np.tanh(0.01 * ((img_array - a_mean) / a_std)) + 1)
        return image

    def mean_norm(self, img_array):
        op_img_array = img_array
        if self.masked:
            # get ROI as a masked np array
            op_img_array = getROIFilter().execute(op_img_array)
            flat_op_img_array = op_img_array.flatten()
            a_mean = np.ma.mean(flat_op_img_array)
            a_min = np.ma.min(flat_op_img_array)
            a_max = np.ma.max(flat_op_img_array)
        else:
            a_mean = np.mean(op_img_array)
            a_min = np.min(op_img_array)
            a_max = np.max(op_img_array)
        image = (img_array - a_mean) / (a_max - a_min)
        return image

    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        img = sample.get('image')
        seg = sample.get('segmentation')

        img_array = sitk.GetArrayFromImage(img)

        if self.norm_method == "min-max":
            image_norm = self.min_max_norm(img_array)
        elif self.norm_method == "z-score":
            image_norm = self.z_score_norm(img_array)
        elif self.norm_method == 'i-scaling':
            image_norm = self.is_norm(img_array)
        elif self.norm_method == 'mean':
            image_norm = self.is_norm(img_array)
        elif self.norm_method == 'tanh':
            image_norm = self.tanH_norm(img_array)
        # elif self.norm_method == "hist_norm":
            # TODO implement method for histogram normalization (requires reference image)
            # image_norm = self.hist_based_normalization(img_array)
        else:
            print('No normalization method was used due to improper specification of method')
            image_norm = img_array

        image_norm = sitk.GetImageFromArray(image_norm)
        image_norm.CopyInformation(img)

        return {'name_img': name_img, 'name_seg': name_seg, 'image': image_norm, 'segmentation': seg}

# TODO create vx spaxing choice based in minimal absolute distance distance between
class ResampleVxSpacing(object):
    def __init__(self, ref_img, reference_center, dimension, default_pixel_value=-1000.0):
        self.ref_img = ref_img
        self.reference_center = reference_center
        self.dimension = dimension
        self.default_pixel_value = default_pixel_value

    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        img = sample.get('image')
        seg = sample.get('segmentation')

        reference_origin = self.ref_img.GetOrigin()

        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(self.dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)

        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(self.dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(
            np.array(transform.GetInverse().TransformPoint(img_center) - self.reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        spaced_image = sitk.Resample(img, self.ref_img, centered_transform,
                                        sitk.sitkBSplineResamplerOrder3, self.default_pixel_value)

        spaced_segmentation = sitk.Resample(seg, self.ref_img, centered_transform,
                                               sitk.sitkNearestNeighbor, 0.0)

        return {'name_img': name_img, 'name_seg': name_seg, 'image': spaced_image, 'segmentation': spaced_segmentation}

class ResampleVxSpacing_no_reference(object):
    def __init__(self, spacingfactor, default_pixel_value=-1000.0):
        self.default_pixel_value = default_pixel_value
        self.spacingfactor = spacingfactor

    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        img = sample.get('image')
        seg = sample.get('segmentation')

        img_size = img.GetSize()
        img_spacing = img.GetSpacing()
        dimension = img.GetDimension()

        reference_origin = np.zeros(dimension)
        reference_direction = np.identity(dimension).flatten()

        reference_spacing = [1*self.spacingfactor] * dimension

        new_reference_img_size = tuple([int(i * (j/k)) for i, j, k in zip(img_size, img_spacing, reference_spacing)])

        new_reference_image = sitk.Image(new_reference_img_size, img.GetPixelIDValue())
        new_reference_image.SetOrigin(reference_origin)
        new_reference_image.SetSpacing(reference_spacing)
        new_reference_image.SetDirection(reference_direction)

        reference_origin = reference_origin
        reference_center = np.array(
            new_reference_image.TransformContinuousIndexToPhysicalPoint(np.array(new_reference_image.GetSize()) / 2.0))

        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)

        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(
            np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        spaced_image = sitk.Resample(img, new_reference_image, centered_transform,
                                     sitk.sitkBSplineResamplerOrder3, self.default_pixel_value)

        spaced_segmentation = sitk.Resample(seg, new_reference_image, centered_transform,
                                            sitk.sitkNearestNeighbor, 0.0)
        print('img_spacing:', img_spacing, 'reference_spacing:', reference_spacing)
        print('img_spacing:', img_size, 'reference_spacing:', new_reference_img_size)
        return {'name_img': name_img, 'name_seg': name_seg, 'image': spaced_image, 'segmentation': spaced_segmentation}

class IntensityClipper(object):
    def __init__(self, lp, hp):
        self.lp = lp
        self.hp = hp

    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        img = sample.get('image')
        seg = sample.get('segmentation')

        img_array = sitk.GetArrayFromImage(img)

        a_min = np.percentile(img_array, self.lp)
        a_max = np.percentile(img_array, self.hp)
        image_clipped = np.clip(img_array, a_min=a_min, a_max=a_max)

        image_clipped = sitk.GetImageFromArray(image_clipped)
        image_clipped.CopyInformation(img)

        return {'name_img': name_img, 'name_seg': name_seg, 'image': image_clipped, 'segmentation': seg}

class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name_img = sample.get('name_img')
        name_seg = sample.get('name_seg')
        img = sample.get('image')
        seg = sample.get('segmentation')

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img_array = sitk.GetArrayFromImage(img)
        seg_array = sitk.GetArrayFromImage(seg)

        img_tensor = torch.from_numpy(img_array)
        seg_tensor = torch.from_numpy(seg_array)

        # Unsqueeze tensor to have proper dimensions
        image = img_tensor.unsqueeze(0)
        segmentation = seg_tensor.unsqueeze(0)
        return {'name_img': name_img, 'name_seg': name_seg, 'image': image, 'segmentation': segmentation}

class getROIFilter():
    "Gets the region of interest (non empty space) from an image."

    def execute(self, img):
        print('checkpoint1')
        # convert img np array back to sitk image
        img = sitk.GetImageFromArray(img)
        # Get the ROI by using otsu based thresholding approach
        # first use straightforward otsu (this doesnt yield perfect results as threshold id too high)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        # otsu_filter.SetNumberOfHistogramBins(32)
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
        print('checkpoint2')
        return mx
