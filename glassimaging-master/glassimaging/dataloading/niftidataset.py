# -*- coding: utf-8 -*-
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import json
import SimpleITK as sitk
from datetime import datetime
import cv2

class NiftiDataset:

    def __init__(self):
        self.df = pd.DataFrame()

        self.ref_img = None
        self.reference_center = None
        self.dimension = None

    """Return an image based on the path as a numpy array

    Not normalized
    """
    def loadImage(self, path):
        # as opposed to the original code by Karin, i'm making use of the sitk imaging library and not nibabel
        img = sitk.ReadImage(path)

        img = sitk.GetArrayFromImage(img)
        return img

    def __len__(self):
        return len(self.df.index)

    """Return an image based on the path as a numpy array

    Normalized using the z-score
    """

    def loadImageNormalized(self, path, technique=None, using_otsu_ROI=False):
        img = sitk.ReadImage(path)
        print("debugging", img.GetPixelIDTypeAsString())
        # returns a sitk image
        img = self.normalize_img(img, technique, using_otsu_ROI)

        return img

    """Return a segmentation based on the path as boolean numpy array
    """
    def loadSegBinarize(self, path):
        img = nib.load(path)
        return img.get_fdata() > 0.5

    def loadSeg(self, path):
        img = nib.load(path)
        return img.get_fdata()


    """Return the full stacked sequence of images of a subject

    Useful for evaluation
    """
    def loadSubjectImages(self, subject, sequences,
                          normalized=True,
                          technique=None,
                          using_otsu_ROI=None,
                          resampling_factor=1):

        if normalized:
            image = [self.loadImageNormalized(self.df.loc[subject][seq],
                                              technique, using_otsu_ROI) for seq in sequences]
        else:
            image = [self.loadImage(self.df.loc[subject][seq]) for seq in sequences]

        image = np.stack(image)
        segmentation = sitk.GetArrayFromImage(sitk.ReadImage(self.df.loc[subject]['seg']))

        # in case an image has a dimension with a size smaller than 112 (default patch size), we add padding to at
        # least be able to fit in the resampling_factor1 patch sizes (also holds for other resampling factors 2,4.
        # Then patch sizes are 56 and 28 respectively)
        # Get original image size
        img_size = image.shape
        seg_size = segmentation.shape
        print(img_size, seg_size)

        # check if img is larger than default patch size for data that has been pre-resampled. ("Res1" or similar must be present in the path if it was pre-resampled otherwise no mention of res is present in the path)
        if "Res1" in self.df.loc[subject][sequences[0]].lower():
            new_img_size = tuple([i if i > 113 else 113 for i in img_size[-3:]])
            new_img_size = img_size[:-3] + new_img_size
            new_seg_size = seg_size[:-3] + new_img_size
        elif "Res2" in self.df.loc[subject][sequences[0]].lower():
            new_img_size = tuple([i if i > 113 else 113 for i in img_size[-3:]])
            new_img_size = img_size[:-3] + new_img_size
            new_seg_size = seg_size[:-3] + new_img_size
        elif "Res4" in self.df.loc[subject][sequences[0]].lower():
            new_img_size = tuple([i if i > 113 else 113 for i in img_size[-3:]])
            new_img_size = img_size[:-3] + new_img_size
            new_seg_size = seg_size[:-3] + new_img_size
        else:
            # if were not working with a pre-resampled dataset
            new_img_size = tuple([i if i > 112 else 112 for i in img_size[-3:]])
            new_img_size = img_size[:-3] + new_img_size
            new_seg_size = seg_size[:-3] + new_img_size
        try:
            # only if the new image size differs we do the change
            if new_img_size != img_size:
                # Here we first create a new image of zeroes and then fill it with the sitk image values
                padded_img_np = np.zeros(new_img_size)
                padded_seg_np = np.zeros(new_img_size)
                padded_img_np[:img_size[0], :img_size[1], :img_size[2], :img_size[3]] = image
                padded_seg_np[:seg_size[0], :seg_size[1], :seg_size[2]] = segmentation
                image = padded_img_np
                segmentation = padded_seg_np
                print(image.shape)
        except:
            print("no extra padding was added to image to comply with patch size")

        # TODO check if resampling should be done using nearest neighbour
        if not resampling_factor == 1:
            image = [self.resample_img2(image[i], sitk.ReadImage(self.df.loc[subject][sequences[i]]), resampling_factor, use_seg=False) for i, seq in enumerate(sequences)]
            segmentation = self.resample_img2(segmentation, sitk.ReadImage(self.df.loc[subject]['seg']), resampling_factor, use_seg=True)

        image = [np.transpose(i) for i in image]
        segmentation = np.transpose(segmentation)
        image = np.stack(image)
        print(image.shape)
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(image[0, :, :, 500])
        # axs[1].imshow(segmentation[:, :, 500])
        #
        # plt.plot()
        # plt.show()

        return image, segmentation

    """Return the full stacked sequence of images of a subject

    Useful for evaluation
    """
    def loadSubjectImagesWithoutSeg(self, subject, sequences,
                          normalized=True,
                          technique=None,
                          using_otsu_ROI=None,
                          resampling_factor=1):

        if normalized:
            image = [self.loadImageNormalized(self.df.loc[subject][seq],
                                              technique, using_otsu_ROI) for seq in sequences]
        else:
            image = [self.loadImage(self.df.loc[subject][seq]) for seq in sequences]

        image = np.stack(image)

        # TODO check if resampling should be done using nearest neighbour
        if not resampling_factor == 1:
            image = [self.resample_img2(image[i], sitk.ReadImage(self.df.loc[subject][sequences[i]]), resampling_factor, use_seg=False) for i, seq in enumerate(sequences)]

        image = [np.transpose(i) for i in image]

        image = np.stack(image)
        print(image.shape)

        return image

    def createCVSplits(self, nsplits):
        self.df['split'] = -1
        split = -1
        for p in np.random.permutation(list(self.df.index)):
            split = (split + 1) % nsplits
            self.df.at[p, 'split'] = split
        self.nsplits = nsplits
        return

    def loadSplits(self, path):
        with open(path, 'r') as file:
            splits = json.load(file)
        ######## Set all patient to split -1, so that only patients in the actual splits file are included
        self.df['split'] = -1
        for i in range(0, len(splits)):
            for p in splits[i]:
                self.df.at[p, 'split'] = i

    def saveSplits(self, loc):
        splits = self.df.split.unique()
        d = [None] * len(splits)
        for i, s in enumerate(splits):
            patients = self.df.loc[self.df['split'] == s].index.values
            d[i] = list(patients)
        path = os.path.join(loc, 'splits.json')
        with open(path, 'w') as file:
            json.dump(d, file, indent=1)

    def getFileName(self, subject, sequence):
        return self.df.at[subject, sequence]

    #function that normalizes a given image based on the technique specified (z-score/robust min-max (i-scaling) etc)
    def normalize_img(self, img, technique, using_otsu_ROI):
        initial_img_np = sitk.GetArrayFromImage(img)
        #TODO check if shape is good
        if using_otsu_ROI:
            # print('Using otsu ROI for normalization')
            img = self.get_ROI_filter(img)
        else:
            img = sitk.GetArrayFromImage(img)

        if technique == 'z-score' or technique is None:
            # use z-scoring
            print('Normalizing usign the z-score')
            # values_nonzero = img[np.nonzero(img)]
            flat_img = img.flatten()
            # getting the non-zero values of the image using the cv2 library (allows for speeduip compared to np)
            values_nonzero = flat_img[np.fliplr(cv2.findNonZero((flat_img > 0).astype(np.uint8)).squeeze())[:,0]]
            mean_nonzero = np.mean(values_nonzero)
            std_nonzero = np.std(values_nonzero)
            if std_nonzero == 0:
                raise ValueError('Standard deviation of image is zero')
            # img[np.nonzero(img)] = (img[np.nonzero(img)] - mean_nonzero) / std_nonzero

            img_n = (initial_img_np - mean_nonzero) / std_nonzero

        elif technique == 'i-scaling':
            # use i-scaling
            print('Normalizing usign intensity scaling normalization')
            # values_nonzero = img[np.nonzero(img)]
            flat_img = img.flatten()
            # getting the non-zero values of the image using the cv2 library (allows for speeduip compared to np)
            values_nonzero = flat_img[np.fliplr(cv2.findNonZero((flat_img > 0).astype(np.uint8)).squeeze())[:,0]]
            LIR = np.percentile(values_nonzero.flatten(), 2)
            HIR = np.percentile(values_nonzero.flatten(), 98)
            # print("intensity norm percentiles:", LIR, HIR)
            # img[np.nonzero(img)] = (img[np.nonzero(img)] - LIR) / (HIR - LIR)
            img_n = (initial_img_np - LIR) / (HIR - LIR)

        # # Used plotting for debugging
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(img[15])
        # axs[0, 1].imshow(img[30])
        # axs[1, 0].imshow(img_n[15])
        # axs[1, 1].imshow(img_n[30])
        # plt.plot()
        # plt.show()

        # returns a masked np array
        return img_n

    # function for getting the ROI filter on the fly
    def get_ROI_filter(self, img):
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

        # # Used for debugging
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(original_img_np[15])
        # axs[1].imshow(mx[15])
        # plt.plot()
        # plt.show()

        return mx

    # Function for applying bias correction on the fly. Advise is no to use this but to do bias correction upfront using the CPU cluster as this method is not accurate enough
    def apply_bias_correction(self, img):
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

        # # used for debugging
        # image_np = sitk.GetArrayFromImage(image)
        # corrected_image_np = sitk.GetArrayFromImage(corrected_image)
        # resampled_exp_logBiasField_np = sitk.GetArrayFromImage(resampled_exp_logBiasField)
        # image_shr_np = sitk.GetArrayFromImage(image_shr)
        # corrected_image_shr_np = sitk.GetArrayFromImage(corrected_image_shr)
        # exp_logBiasField_np = sitk.GetArrayFromImage(exp_logBiasField)
        #
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

        return corrected_image

    # function for resampling on the fly. Only use this in the case that dataset itself has not been resampled
    def resample_img2(self, image, reference_img, resampling_factor, use_seg=False):
        # get image variables
        img_spacing = reference_img.GetSpacing()
        img_direction = reference_img.GetDirection()
        img_origin = reference_img.GetOrigin()
        img_size = reference_img.GetSize()
        img_pixelIDValue = reference_img.GetPixelIDValue()

        # Define new size and spacing based on resampling factor
        new_img_size = tuple(int(i / resampling_factor) for i in img_size)
        new_img_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_img_size, img_size, img_spacing)]

        # Create an image to resample to (your target image so to say) and set the size, ID value and spacing
        resample_to_this_image = sitk.Image(*new_img_size, img_pixelIDValue)

        resample_to_this_image.SetSpacing(new_img_spacing)

        image_to_resample = sitk.GetImageFromArray(image)

        # Depending on whether we are dealing with a segmentation or an image that needs resampling we use one of the following two operations
        # The difference is in the fact that for images we use the 3rd order spline interpolation and for segmentations we use nearest neighbor
        if not use_seg:
            resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkBSplineResamplerOrder3)
        else:
            resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkNearestNeighbor)

        resampled_img_np = sitk.GetArrayFromImage(resampled_img)

        return resampled_img_np

    # function that creates a reference domain. Can be used in resampling but is not
    # def __create_reference_domain(self, dataset, isotropic: bool = False, vx_spacing: str = 'median'):
    #     # dataset is a dataframe with all the names of usable data
    #     img_0 = sitk.ReadImage(dataset[dataset.columns[0]][0])
    #     # will most likely be 3D
    #     dimension = img_0.GetDimension()
    #
    #     # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    #     reference_physical_size = np.zeros(dimension)
    #     biggest_img_size = 0
    #     spacing_list = []
    #     sizes_list = []
    #
    #     for index, subject in enumerate(dataset[dataset.columns[0]]):
    #         img = sitk.ReadImage(subject)
    #         reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
    #                                       zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    #         biggest_img_size = max(img.GetSize()) if max(img.GetSize()) > biggest_img_size else biggest_img_size
    #         spacing_list.append(img.GetSpacing())
    #         sizes_list.append(img.GetSize())
    #
    #     sizes_list.sort()
    #
    #     spacing_list.sort()
    #
    #
    #     if isotropic:
    #         # The first possibility is that you want isotropic pixels, if so you can specify the image size for one of
    #         # the axes and the others are determined by this choice. Below we choose to set the x axis to the biggest
    #         # image size and the spacing accordingly.
    #         if vx_spacing == 'median':
    #             reference_spacing = [spacing_list[int(len(spacing_list) / 2)][0]] * dimension
    #         elif vx_spacing == 'mean':
    #             reference_spacing = list(map(lambda y: math.ceil(sum(y) / float(len(y))), zip(*spacing_list)))[0] * dimension
    #         elif vx_spacing == 'min':
    #             reference_spacing = [spacing_list[0][0]] * dimension
    #         elif vx_spacing == 'max':
    #             reference_spacing = [spacing_list[-1][0]] * dimension
    #         elif vx_spacing == 'development':
    #             # print('prev spc', spacing_list[-1][0])
    #             multiplier = (1, 4, 4)
    #             reference_spacing = [tuple(i * j for i, j in zip(multiplier, spacing_list[-1][0]))] * dimension
    #             print('new spc', reference_spacing)
    #         else:
    #             # Just use the median spacing
    #             reference_spacing = [spacing_list[int(len(spacing_list) / 2)][0]] * dimension
    #         # print('ref spacing:', reference_spacing)
    #         reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                           zip(reference_physical_size, reference_spacing)]
    #         reference_size = self.adjust_reference_size(reference_size)
    #         # print(reference_size, reference_size_x)
    #     else:
    #         # in the case that we do not want to use isotropic spacing we come here and set the spacing values according to the set parameters
    #         if vx_spacing == 'median':
    #             reference_spacing = spacing_list[int(len(spacing_list) / 2)]
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #         elif vx_spacing == 'mean':
    #             reference_spacing = list(map(lambda y: math.ceil(sum(y) / float(len(y))), zip(*spacing_list)))
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #         elif vx_spacing == 'min':
    #             reference_spacing = spacing_list[0]
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #         elif vx_spacing == 'max':
    #             reference_spacing = spacing_list[-1]
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #         elif vx_spacing == 'development':
    #             #The development spacing is used to do test runs of the code. As resampling may take some time, here we use a spacing factor of 4 to reduce the data needing resampling.
    #             print('prev spc', spacing_list[-1])
    #             multiplier = (4, 4, 1)
    #             reference_spacing = tuple(i * j for i, j in zip(multiplier, spacing_list[-1]))
    #             print('new spc', reference_spacing)
    #
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #
    #         else:
    #             # Just use the median spacing
    #             reference_spacing = spacing_list[int(len(spacing_list) / 2)]
    #             reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
    #                               zip(reference_physical_size, reference_spacing)]
    #             reference_size = self.adjust_reference_size(reference_size)
    #
    #         print('      Reference spacing will be:', reference_spacing)
    #         print('      Reference size will be:', reference_size)
    #
    #     reference_image = sitk.Image(reference_size, img_0.GetPixelIDValue())
    #     # reference_image.SetOrigin(reference_origin)
    #     reference_image.SetSpacing(reference_spacing)
    #     # reference_image.SetDirection(reference_direction)
    #
    #     # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    #     # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    #     # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    #     # spacing will not yield the correct coordinates resulting in a long debugging session.

    #     reference_center = np.array(
    #         reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    #
    #     return reference_image, reference_center, dimension

    def adjust_reference_size(self, reference_size):
        adjusted_reference_size = []
        for x in reference_size:
            adjusted_reference_size.append(math.ceil(x / 16.0) * 16)
        return adjusted_reference_size

class ResampleVxSpacing(object):
    def __init__(self, ref_img, reference_center, dimension):
        self.ref_img = ref_img
        self.reference_center = reference_center
        self.dimension = dimension

    def run(self, img, seg):

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
                                        sitk.sitkBSplineResamplerOrder3, 0.0)

        spaced_segmentation = sitk.Resample(seg, self.ref_img, centered_transform,
                                            sitk.sitkNearestNeighbor, 0.0)

        return spaced_image, spaced_segmentation
