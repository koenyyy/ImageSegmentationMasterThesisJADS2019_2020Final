import os
import SimpleITK as sitk
import numpy as np


def correct_dataset():
    dataset_loc = '/media/data/kvangarderen/BTD'
    save_loc = 'media/data/kderaad/BTD_N4BC'

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
