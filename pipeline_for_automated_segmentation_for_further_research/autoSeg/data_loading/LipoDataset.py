import torch
from torch.utils.data import Dataset
import os
import numpy as np
import SimpleITK as sitk


class LipoDataset(Dataset):
    """Lipo segmentation dataset."""

    def __init__(self, root_dir='..//data_for_testing//LipoData', transform=None, seg_to_use=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Listing all the names of the images and segmentations
        self.dataset_name = 'LipoData'
        self.root_dir = root_dir
        self.lstFilesGz = []  # create an empty list with images
        self.lstFilesGzSeg = []  # create an empty list with segmentations
        names = ['anthony', 'melissa']  # specify the names of person who conducted the segmentation

        for dirName, subdirList, fileList in os.walk(self.root_dir):
            for filename in fileList:
                if ".gz" in filename.lower():  # check whether the file's .gz
                    # check wether a file has a segmentation from a specific person
                    if any(name in filename.lower() for name in names):
                        self.lstFilesGz.append(os.path.join(dirName, 'image.nii.gz'))
                        self.lstFilesGzSeg.append(os.path.join(dirName, filename))

        self.transform = transform

    def _loadImage(self, path):
        image = sitk.ReadImage(path)
        return image

    def _loadSeg(self, path):
        segmentation = sitk.ReadImage(path)
        return segmentation

    def __len__(self):
        return len(self.lstFilesGz)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.lstFilesGz[idx]
        seg_name = self.lstFilesGzSeg[idx]

        image = self._loadImage(self.lstFilesGz[idx])
        segmentation = self._loadSeg(self.lstFilesGzSeg[idx])

        sample = {'name_img': img_name, 'name_seg': seg_name, 'image': image, 'segmentation': segmentation}
        if self.transform:
            sample = self.transform(sample)

        return sample
