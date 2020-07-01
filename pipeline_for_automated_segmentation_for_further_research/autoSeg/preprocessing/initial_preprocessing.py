import json
import math
import os
import importlib


# TODO make dynamic for load data
from autoSeg.preprocessing.Transform import OtsuCrop, ResampleVxSpacing, ResampleVxSpacing_no_reference, N4BiasCorrection, Normalize, IntensityClipper

import SimpleITK as sitk
from SimpleITK import N4BiasFieldCorrection, GetImageFromArray
import numpy as np
from torchvision import transforms
import sys


class InitialPreprocessor(object):
    def __init__(self, INPUT_data_dir, config_file):
        self.INPUT_data_dir = INPUT_data_dir
        self.INTERMEDIATE_data_dir = os.path.join(sys.path[0], 'data', 'intermediate_data')
        self.PREPROCESSED_data_dir = os.path.join(sys.path[0], 'data', 'preprocessed_data')

        # Loading the json data from specified config file
        config_file = os.path.join(sys.path[0], 'autoSeg', 'config', config_file)
        with open(config_file, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.config_file = config_json_data

        self.dataset_import = getattr(importlib.import_module('autoSeg.data_loading.{0:s}'.format(self.config_file['dataset'][0])), '{0:s}'.format(self.config_file['dataset'][1]))

    def run(self):
        # Getting settings from config file
        isotropic = self.config_file.get('isotropic')
        vx_spacing = self.config_file.get('vxspacing')
        spacingfactor = self.config_file.get('vxspacingfactor')

        # Load the inital base dataset and perform selected options
        print('   Loading initial dataset')
        dataset = self.load_data(self.INPUT_data_dir)

        dataset_origin_path = self.INTERMEDIATE_data_dir

        # Logging what transforms the intermediate dataset will have at the end
        dataset_state = {}
        dataset_state['biascorrection'] = self.config_file.get('biascorrection')
        dataset_state['resampling'] = self.config_file.get('resampling')
        dataset_state['isotropic'] = self.config_file.get('isotropic')
        dataset_state['vxspacing'] = self.config_file.get('vxspacing')

        # TODO zorgen dat croppen zonder resamplen kan
        # Check if there is a previous dataset
        print('   Checking for previously preprocessed datasets')
        json_check_path = os.path.join(self.INTERMEDIATE_data_dir, 'resampling.json')
        if os.path.isfile(json_check_path):
            # Below we define what to do when the intermediate dataset is already in a certain state
            # (e.g. biascoorection and no resampling)
            if not self.get_previous_config('biascorrection') and self.get_previous_config('resampling'):
                bc = self.config_file.get('biascorrection')
                res = self.config_file.get('resampling')
                if not bc and not res:
                    dataset_origin_path = self.INPUT_data_dir
                elif bc and not res:
                    dataset_origin_path = self.INPUT_data_dir
                elif not bc and res:
                    if self.config_file.get('isotropic') == self.get_previous_config('isotropic') and \
                            self.config_file.get('vxspacing') == self.get_previous_config('vxspacing'):
                        self.config_file['cropping'] = False
                        self.config_file['resampling'] = False
                    else:
                        dataset_origin_path = self.INPUT_data_dir
                elif bc and res:
                    self.config_file['cropping'] = False
                    self.config_file['resampling'] = False

            elif self.get_previous_config('biascorrection') and not self.get_previous_config('resampling'):
                bc = self.config_file.get('biascorrection')
                res = self.config_file.get('resampling')
                if not bc and not res:
                    dataset_origin_path = self.INPUT_data_dir
                elif bc and not res:
                    self.config_file['biascorrection'] = False
                elif not bc and res:
                    dataset_origin_path = self.INPUT_data_dir
                elif bc and res:
                    self.config_file['biascorrection'] = False

            elif self.get_previous_config('biascorrection') and self.get_previous_config('resampling'):
                dataset_origin_path = self.INPUT_data_dir
        else:
            # In case no prev dataset exists we use the input dataset
            dataset_origin_path = self.INPUT_data_dir

       # If resampling is set to true in the config we perform resampling
        if self.config_file.get('resampling'):
            print('   Resampling dataset is needed')
            # Load the dataset and perform resampling on the voxel spacing
            ref_img, reference_center, dimension = self.__create_reference_domain(dataset,
                                                                                  isotropic=isotropic,
                                                                                  vx_spacing=vx_spacing,
                                                                                  spacingfactor=spacingfactor)
            print('   Dataset will be resampled to shape:', ref_img.GetSize(), 'and spacing:', vx_spacing,
                  'with default value:', self.config_file.get('default_pixel_value'))
            if not self.config_file.get('resample_without_reference'):
                resample_obj = ResampleVxSpacing(ref_img, reference_center, dimension,
                                                 default_pixel_value=self.config_file.get('default_pixel_value'))
            else:
                # Here, only isotropic spacing is possible
                resample_obj = ResampleVxSpacing_no_reference(spacingfactor=spacingfactor, default_pixel_value=self.config_file.get('default_pixel_value'))
        else:
            resample_obj = None
            self.config_file['cropping'] = False
            self.config_file['resampling'] = False

        transforms_list = self.get_transforms_list(resample_obj)
        print('   Needed transformations are:', transforms_list)
        # Here we cut the list of transformations in two to ensure that we do the bias correction and resampling first,
        # save it in the intermediate_data folder and then use it for further transformations
        if resample_obj in transforms_list:
            i = transforms_list.index(resample_obj)
            transforms_list1 = transforms_list[:i + 1]
            transforms_list2 = transforms_list[i + 1:]

            print('   Loading initial dataset for resampling')
            dataset_transf1 = self.load_data(dataset_origin_path, transforms_list1)
            print('   Saving resampled dataset:')
            self.save(dataset_transf1, self.INTERMEDIATE_data_dir)

            print('   Loading resampled dataset for further preprocessing')
            dataset_transf = self.load_data(self.INTERMEDIATE_data_dir, transforms_list2)
            print('   Saving fully preprocessed dataset:')
            self.save(dataset_transf, self.PREPROCESSED_data_dir)
        else:
            # If there is no resampling needed we dont slice the list
            print('   Loading older resampled dataset for further preprocessing')
            dataset_transf = self.load_data(self.INTERMEDIATE_data_dir, transforms_list)
            print('   Saving fully preprocessed dataset:')
            self.save(dataset_transf, self.PREPROCESSED_data_dir)

        # Here we store the executed resampling steps in a file that we can check in later uses.
        json_check_path = os.path.join(self.INTERMEDIATE_data_dir, 'resampling.json')
        with open(json_check_path, 'w') as outfile:  # Use file to refer to the file object
            json.dump(dataset_state, outfile)

        # check whether the dataset is valid and can be used in the training procedure. ssp stands for
        # sizes, spacing and pooling
        is_valid_dataset, ssp = self.dataset_valid(dataset_transf)
        if is_valid_dataset:
            print('   Dataset valid for training')
        else:
            raise ValueError('Dataset is not valid for training. Sizes, spacing and pooling are:', ssp)

    def load_data(self, data_dir, transforms_list: list = None):
        if transforms_list is None:
            transforms_list = []
        transform = transforms.Compose(transforms_list)
        # data_set = eval(self.config_file['dataset'][1])(root_dir=data_dir, transform=transform)
        data_set = self.dataset_import(root_dir=data_dir, transform=transform)

        return data_set

    def get_transforms_list(self, resampleObj=None):
        transforms_list = []
        # If config says we need to do a transformation, we add this to the list
        transf_options_dict = {
            "biascorrection": N4BiasCorrection(),
            "cropping": OtsuCrop(),
            "resampling": resampleObj,
            "intensityclipping": IntensityClipper(self.config_file.get('intensityclippingvalues')[0],
                                                  self.config_file.get('intensityclippingvalues')[1]),
            "normalization": Normalize(norm_method=self.config_file.get('normalizationmethod'),
                                       lir=self.config_file.get('lirHir')[0],
                                       hir=self.config_file.get('lirHir')[1],
                                       masked=self.config_file.get('masked'))

        }

        for transf_option in transf_options_dict:
            if self.config_file.get('{}'.format(transf_option)):
                transforms_list.append(transf_options_dict.get(transf_option))

        return transforms_list

    def save(self, dataset, OUTPUT_data_dir):
        for subject in dataset:
            name_img = subject.get('name_img')
            name_seg = subject.get('name_seg')
            img = subject.get('image')
            seg = subject.get('segmentation')

            # create path to store the images. The path has the same folder structure as initial dataset
            file_struct_img = name_img.split('_data')[1].split(os.sep)[:-1]
            file_struct_seg = name_seg.split('_data')[1].split(os.sep)[:-1]


            folder_path_img = os.path.join(OUTPUT_data_dir, *file_struct_img)
            folder_path_seg = os.path.join(OUTPUT_data_dir, *file_struct_seg)

            # Check if the path already exists. If not make the path
            print('      Saving to: ', folder_path_img, 'and', folder_path_seg)
            if not os.path.isdir(folder_path_img):
                os.makedirs(folder_path_img)
            if not os.path.isdir(folder_path_seg):
                os.makedirs(folder_path_seg)
            # check if images are already gzipped, if not do it anyways
            save_img_name = name_img.split(os.path.sep)[-1] + '.gz' if not '.gz' in name_img.split(os.path.sep)[-1] else name_img.split(os.path.sep)[-1]
            save_seg_name = name_seg.split(os.path.sep)[-1] + '.gz' if not '.gz' in name_seg.split(os.path.sep)[-1] else name_seg.split(os.path.sep)[-1]
            img_file_path = os.path.join(folder_path_img, save_img_name)
            seg_file_path = os.path.join(folder_path_seg, save_seg_name)
            sitk.WriteImage(img, str(img_file_path))
            sitk.WriteImage(seg, str(seg_file_path))

    def get_previous_config(self, param: str):
        prev_conf = False

        # construct path to file
        json_check_path = os.path.join(self.INTERMEDIATE_data_dir, 'resampling.json')

        # File might not exist, in that case we use the exception
        try:
            with open(json_check_path, 'r') as infile:  # Use file to refer to the file object
                data = json.load(infile)
                prev_conf = data.get(param)
        except IOError:
            pass
        return prev_conf

    def dataset_valid(self, dataset):
        size_list = []
        vx_spacing_list = []

        for subject in dataset:
            img = subject.get('image')

            size_list.append(img.GetSize())
            vx_spacing_list.append(img.GetSpacing())

        # Since we do 4 max pooling operations with filtersize 2x2(x2) we need to get an amount of layers that is
        # divisible by 2x2x2x2=16
        same_sizes = size_list[:-1] == size_list[1:]
        same_spacing = vx_spacing_list[:-1] == vx_spacing_list[1:]
        # TODO get the pooling size from the training config file in order to determine if pooling is possible
        max_pooling_possible = None

        for (x, y, z) in size_list:
            if x % 16 == 0 and y % 16 == 0 and z % 16 == 0:
                max_pooling_possible = True
            else:
                max_pooling_possible = False
                break
        return same_sizes and same_spacing and max_pooling_possible, (same_sizes, same_spacing, max_pooling_possible)

    def adjust_reference_size(self, referene_size):
        adjusted_reference_size = []
        for x in referene_size:
            adjusted_reference_size.append(math.ceil(x / 16.0) * 16)
        return adjusted_reference_size

    def __create_reference_domain(self, dataset, isotropic: bool = True, vx_spacing: str = 'median',
                                  spacingfactor: int = 1):
        # In Lipo Data this will be 3D
        dimension = dataset[0].get('image').GetDimension()

        # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
        reference_physical_size = np.zeros(dimension)
        biggest_img_size = 0
        spacing_list = []
        sizes_list = []

        for index, subject in enumerate(dataset):
            img = subject.get('image')
            reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                          zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
            biggest_img_size = max(img.GetSize()) if max(img.GetSize()) > biggest_img_size else biggest_img_size
            spacing_list.append(img.GetSpacing())
            sizes_list.append(img.GetSize())

        # TODO sort for indiv tuple values, not tuples as a whole
        sizes_list.sort()
        # print('sizelist:', sizes_list)
        # print('spacing list:', spacing_list)
        spacing_list.sort()
        # print('spacing list ordered:', spacing_list)
        # print('median spacing: ', spacing_list[in  t(len(spacing_list) / 2)])
        # Create the reference image with a zero origin, identity direction cosine matrix and dimension
        reference_origin = np.zeros(dimension)
        reference_direction = np.identity(dimension).flatten()

        # TODO change voxel spacing from smallest to mean
        # TODO implement 'smart spacing' option that minimizes overall distance to all points
        if isotropic:
            # The first possibility is that you want isotropic pixels, if so you can specify the image size for one of
            # the axes and the others are determined by this choice. Below we choose to set the x axis to the biggest
            # image size and the spacing accordingly.
            if vx_spacing == 'median':
                reference_spacing = [(spacing_list[int(len(spacing_list) / 2)][0])*spacingfactor] * dimension
            elif vx_spacing == 'mean':
                reference_spacing = list(map(lambda y: math.ceil(sum(y) / float(len(y))), zip(*spacing_list)))[0] * dimension
            elif vx_spacing == 'min':
                reference_spacing = [spacing_list[0][0]*spacingfactor] * dimension
            elif vx_spacing == 'max':
                reference_spacing = [spacing_list[-1][0]*spacingfactor] * dimension
            elif vx_spacing == 'one':
                reference_spacing = [1*spacingfactor] * dimension
            elif vx_spacing == 'development':
                print('prev spc', spacing_list[-1][0])
                multiplier = (1, 4, 4)
                reference_spacing = [tuple(i * j for i, j in zip(multiplier, spacing_list[-1][0]))] * dimension
                print('new spc', reference_spacing)
            else:
                # Just use the median spacing
                reference_spacing = [spacing_list[int(len(spacing_list) / 2)][0]] * dimension
            # print('ref spacing:', reference_spacing)
            reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                              zip(reference_physical_size, reference_spacing)]
            reference_size = self.adjust_reference_size(reference_size)
            # print(reference_size, reference_size_x)
        else:
            # Select arbitrary number of pixels per dimension, smallest size that yields desired results
            # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
            # often result in non-isotropic pixel spacing. Here we have chosen to use the largest img size to
            # prevent loss of valuable information. Using this will make the image anisotropic. The effect
            # will most likely be seen in the 3rd dimension / over the slices.
            if vx_spacing == 'median':
                reference_spacing = spacing_list[int(len(spacing_list) / 2)]
                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)
            elif vx_spacing == 'mean':
                reference_spacing = list(map(lambda y: math.ceil(sum(y) / float(len(y))), zip(*spacing_list)))
                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)
            elif vx_spacing == 'min':
                reference_spacing = spacing_list[0]
                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)
            elif vx_spacing == 'max':
                reference_spacing = spacing_list[-1]
                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)
            elif vx_spacing == 'development':
                print('prev spc', spacing_list[-1])
                multiplier = (4, 4, 1)
                reference_spacing = tuple(i * j for i, j in zip(multiplier, spacing_list[-1]))
                print('new spc', reference_spacing)

                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)

            else:
                # Just use the median spacing
                reference_spacing = spacing_list[int(len(spacing_list) / 2)]
                reference_size = [int(phys_sz / (spc) + 1) for phys_sz, spc in
                                  zip(reference_physical_size, reference_spacing)]
                reference_size = self.adjust_reference_size(reference_size)

            print('      Reference spacing will be:', reference_spacing)
            print('      Reference size will be:', reference_size)

        reference_image = sitk.Image(reference_size, dataset[0].get('image').GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
        # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
        # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
        # spacing will not yield the correct coordinates resulting in a long debugging session.
        # TODO implement function that takes into account the direction cosine of all images.
        reference_center = np.array(
            reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

        return reference_image, reference_center, dimension
