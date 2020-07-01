from autoSeg.data_loading.BTDDatasetKoen import BTDDataset
from autoSeg.preprocessing.Transform import N4BiasCorrection
import os  
import SimpleITK as sitk


def save(dataset, OUTPUT_data_dir):
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


        img_file_path = os.path.join(folder_path_img, name_img.split(os.path.sep)[-1])
        seg_file_path = os.path.join(folder_path_seg, name_seg.split(os.path.sep)[-1])
        sitk.WriteImage(img, str(img_file_path))
        sitk.WriteImage(seg, str(seg_file_path))


data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"
n4bc_data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/n4bc_dataset"

dataset_transf = BTDDataset(data_dir, transform=N4BiasCorrection(), seg_to_use=[True, True, True])
# save the image to n4bc folder
print('   Saving resampled dataset to:', n4bc_data_dir)
save(dataset_transf, n4bc_data_dir)
