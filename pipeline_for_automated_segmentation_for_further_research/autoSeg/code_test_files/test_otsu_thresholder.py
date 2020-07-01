import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from autoSeg.data_loading.LipoDataset import LipoDataset
from autoSeg.data_loading.LiTSDataset import LiTSDataset
from autoSeg.preprocessing.Transform import N4BiasCorrection

data_dir_lits = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
data_dir_lipo = "C:/Users/s145576/Documents/.Koen de Raad/year19.20/Thesis/Erasmus MC/Data/SingleLipoSample"


dataset_lits = LiTSDataset(data_dir_lits, transform=N4BiasCorrection(), seg_to_use=[True, True, True])
dataset_lipo = LipoDataset(data_dir_lipo, transform=N4BiasCorrection(), seg_to_use=[True, True, True])

class getROIFilter():
    "Gets the region of interest (non empty space) from an image."

    def execute(self, img):
        print('##########', img.GetPixelIDTypeAsString())
        # img = sitk.Cast(img, sitk.sitkUInt16)

        print('checkpoint1')
        if isinstance(img, sitk.Image):
            print('is already of type image')
        else:
            print('converting to image')
            # convert img np array back to sitk image
            img = sitk.GetImageFromArray(img)

        # Get the ROI by using otsu based thresholding approach
        # first use straightforward otsu (this doesnt yield perfect results as threshold id too high)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        # otsu_filter.SetNumberOfHistogramBins(32)
        ROI = otsu_filter.Execute(img)
        print(otsu_filter.GetThreshold())

        # locate the negated parts of the image
        otsu_filter2 = sitk.OtsuThresholdImageFilter()
        otsu_filter2.SetInsideValue(1)
        otsu_filter2.SetOutsideValue(0)
        ROI_neg = otsu_filter2.Execute(img)
        mask_filter = sitk.MaskImageFilter()
        masked_ROI_neg = mask_filter.Execute(img, ROI_neg)
        print(otsu_filter2.GetThreshold())

        # use otsu again on the negated part to find values that are close to empty space but are not empty space
        otsu_filter3 = sitk.OtsuThresholdImageFilter()
        otsu_filter3.SetInsideValue(0)
        otsu_filter3.SetOutsideValue(1)
        ROI_addition = otsu_filter3.Execute(masked_ROI_neg)
        print(otsu_filter3.GetThreshold())

        # Add two passes of otsu together here
        combine_filter = sitk.AddImageFilter()
        combined_filters = combine_filter.Execute(ROI, ROI_addition)

        combined_filters_np = sitk.GetArrayFromImage(combined_filters)

        original_img_np = sitk.GetArrayFromImage(img)
        # use a numpy mask for excluding irrelevant data
        mx = np.ma.masked_array(original_img_np, mask=np.logical_not(combined_filters_np))
        return mx


class Run_me(object):
    def run(self):
        path_lits = 'C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data/volume-0.nii/volume-0.nii'
        path_btd = 'C:/Users/s145576/Documents/.Koen de Raad/year19.20/Thesis/Erasmus MC/Data/BTD/BTD-0002/2310.nii.gz'
        path_lipo = 'C:/Users/s145576/Documents/.Koen de Raad/year19.20/Thesis/Erasmus MC/Data/AllLipoData/LipoRadiomics-001/LipoRadiomics-001/4/image.nii.gz'

        img_lits = sitk.ReadImage(path_lits)
        img_lipo = sitk.ReadImage(path_lipo)
        img_btd = sitk.ReadImage(path_btd)

        img_list = [img_lits, img_lipo, img_btd]

        ROI_img_list = []
        for img in img_list:
            ROI_img_list.append(getROIFilter().execute(img))

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(ROI_img_list[0][30])
        axs[1].imshow(ROI_img_list[1][30])
        axs[2].imshow(ROI_img_list[2][30])

        plt.plot()
        plt.show()

    def run2(self):
        ROI_img_list = []
        counter = 0

        for sample in dataset_lits:
            img = sample.get('image')
            ROI_img_list.append(getROIFilter().execute(img))
            counter += 1
            if counter == 2:
                counter = 0
                break

        for sample in dataset_lipo:
            img = sample.get('image')
            ROI_img_list.append(getROIFilter().execute(img))
            counter += 1
            if counter == 2:
                counter = 0
                break

        print(len(ROI_img_list))
        fig, axs = plt.subplots(4, 1)
        axs[0].imshow(ROI_img_list[0][0])
        axs[1].imshow(ROI_img_list[1][30])
        axs[2].imshow(ROI_img_list[2][30])
        axs[3].imshow(ROI_img_list[2][30])

        plt.plot()
        plt.show()

Run_me().run2()
