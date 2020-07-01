import importlib
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import SimpleITK as sitk
from torchvision import transforms
from autoSeg.data_loading.LipoDataset import LipoDataset
import sys
from tqdm import tqdm

class FeatureExtractorObject(object):
    def __init__(self, INPUT_data_dir, config_file):
        self.INPUT_data_dir = INPUT_data_dir

        # Loading the json data from specified config file
        config_file = os.path.join(sys.path[0], 'autoSeg', 'config', config_file)
        with open(config_file, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.config_file = config_json_data

        self.dataset_import = getattr(
            importlib.import_module('autoSeg.data_loading.{0:s}'.format(self.config_file['dataset'][0])),
            '{0:s}'.format(self.config_file['dataset'][1]))

    def run(self):
        if self.already_feature_extraction():
            print('   Features of DS were already extracted')
            return

        # Getting settings from config file
        dataset = self.load_data(self.INPUT_data_dir)

        meta_data = {}

        subj_dimension = []
        subj_spacing = []
        subj_size = []

        img_volume = []
        seg_volume = []
        ROI_volume = []

        # intensities are denoted as {'mean':x, 'median':y, 'mode':z, 'std':f, 'var':g, 'd_range':h, 'IQR':p, 'modality':'bimodal', 'skewness':q, 'kurtosis':r, }
        img_intensity = []
        seg_intensity = []
        ROI_intensity_descriptor = []

        for index, subject in tqdm(enumerate(dataset)):
            # Below, the ROI or Region of interest comprise the voxels that are not 'empty space'
            img = subject.get('image')
            seg = subject.get('segmentation')

            subj_dimension.append(img.GetDimension())
            subj_spacing.append(img.GetSpacing())
            subj_size.append(img.GetSize())

            # Getting the region of interest
            ROI = self.get_ROI_filter(img)

            # NIET VERWIJDEREN HANDIG MET DEBUGGEN
            # # visualisatoin of the images before and after ROI extraction
            # ROI_np = sitk.GetArrayFromImage(ROI)
            # img_np = sitk.GetArrayFromImage(img)
            # plt.imshow(ROI_np[8], cmap='gray', alpha=1, interpolation='none')
            # plt.show()
            # plt.imshow(img_np[8], cmap='gray', alpha=1, interpolation='none')
            # plt.show()

            # print(np.max(sitk.GetArrayFromImage(seg)))
            # TODO make sure that only the ROI is used for calculations and not the whole image. same for seg
            img_volume.append(self.calc_image_volume(img, use_filter=False))
            seg_volume.append(self.calc_image_volume(seg, use_filter=True))
            ROI_volume.append(self.calc_image_volume(ROI, use_filter=True))

            # intensities are denoted as {'mean':x, 'median':y, 'mode':z, 'std':f, 'var':g, 'd_range':h, 'IQR':p, 'modality':'bimodal', 'skewness':q, 'kurtosis':r, }
            img_intensity.append(self.calc_intensity_stats(img, img, use_filter=False))
            seg_intensity.append(self.calc_intensity_stats(seg, img, use_filter=True))
            ROI_intensity_descriptor.append(self.calc_intensity_stats(ROI, img, use_filter=True))

            meta_data['subj_dimension'] = subj_dimension
            meta_data['subj_spacing'] = subj_spacing
            meta_data['subj_size'] = subj_size
            meta_data['img_volume'] = img_volume
            meta_data['seg_volume'] = seg_volume
            meta_data['ROI_volume'] = ROI_volume
            meta_data['img_intensity'] = img_intensity
            meta_data['seg_intensity'] = seg_intensity
            meta_data['ROI_intensity_descriptor'] = ROI_intensity_descriptor

        print('   meta data:', meta_data)

        self.save_meta_data(meta_data)


    def load_data(self, data_dir, transforms_list: list = None):
        if transforms_list is None:
            transforms_list = []
        transform = transforms.Compose(transforms_list)
        data_set = self.dataset_import (root_dir=data_dir, transform=transform)

        return data_set

# TODO check if you can make it a bit less omslachtig
    def get_ROI_filter(self, img):
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
        # mask_filter = sitk.MaskImageFilter()
        # final_ROI = mask_filter.Execute(img, combined_filters)
        # print(np.max(sitk.GetArrayFromImage(combined_filters)))
        return combined_filters

    def calc_image_volume(self, img, use_filter=False):
        if not use_filter:
            # get spacing and amount of voxels in the image
            spc = img.GetSpacing()
            pxnr = img.GetNumberOfPixels()

            # calculate the colume of a single voxel
            voxvol = 1
            for i in spc:
                voxvol = voxvol * i

            # the total image volume
            img_volume = pxnr * voxvol
        elif use_filter:
            # get spacing and amount of voxels in the image
            spc = img.GetSpacing()
            img_np = sitk.GetArrayFromImage(img)

            # get the amount of non zero voxels from the masks
            non_zero_px = np.count_nonzero(img_np)

            # calculate the colume of a single voxel
            voxvol = 1
            for i in spc:
                voxvol = voxvol * i

            # the total image volume
            img_volume = non_zero_px * voxvol

        return img_volume
    def calc_intensity_stats(self, img, original_img, use_filter=False):
        # format to return is: {'mean': x, 'median': y, 'mode': z, 'std': f, 'var': g, 'd_range': h, 'IQR': p, 'modality': 'bimodal', 'skewness': q, 'kurtosis': r, }
        img_np = sitk.GetArrayFromImage(img)
        flat_img_np = img_np.flatten()

        if not use_filter:
            mean = np.mean(flat_img_np)
            median = np.median(flat_img_np)
            mode = float(stats.mode(flat_img_np)[0])
            std = np.std(flat_img_np)
            var = np.var(flat_img_np)
            d_range = float(np.min(flat_img_np)), float(np.max(flat_img_np))
            IQR = list(np.percentile(flat_img_np, [75 ,25])) # gives q75, q25
            # modality = nope not yet (too complex to implement)
            skewness = stats.skew(flat_img_np, bias=True)
            kurtosis = stats.kurtosis(flat_img_np, bias=True)
        elif use_filter:
            original_img_np = sitk.GetArrayFromImage(original_img)
            # use a numpy mask for excluding irrelevant data
            mx = np.ma.masked_array(original_img_np, mask=np.logical_not(img_np))
            flat_mx_np = mx.flatten()
            mean = np.ma.mean(flat_mx_np)
            median = np.ma.median(flat_mx_np)
            mode = float(stats.mode(flat_mx_np)[0])
            std = np.ma.std(flat_mx_np)
            var = np.ma.var(flat_mx_np)
            d_range = float(np.ma.min(flat_mx_np)), float(np.ma.max(flat_mx_np))
            IQR = [float(i) for i in np.percentile(flat_mx_np, [75, 25])]  # gives q75, q25
            # modality = nope not yet (too complex to implement)
            skewness = stats.skew(flat_mx_np, bias=True)
            kurtosis = stats.kurtosis(flat_mx_np, bias=True)

        intensity_stats = {'mean': float(mean), 'median': float(median), 'mode': float(mode), 'std': float(std), 'var': float(var),
                           'd_range': d_range, 'IQR': IQR, 'skewness': float(skewness), 'kurtosis': float(kurtosis)}

        return intensity_stats

    def save_meta_data(self, meta_data):
        print('   Saving dataset meta data')
        # getting to the right folder and trained network
        meta_data_loc = os.path.join(sys.path[0], 'data', 'input_data', 'dataset_meta_data.json')

        with open(meta_data_loc, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)

    def already_feature_extraction(self):
        meta_data_loc = os.path.join(sys.path[0], 'data', 'input_data', 'dataset_meta_data.json')
        try:
            with open(meta_data_loc) as f:
                return True
        except FileNotFoundError:
            return False
