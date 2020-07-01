import torch
from torch.utils.data import Dataset
import os
import numpy as np
import SimpleITK as sitk


class LiTSDataset(Dataset):
    """LiTS  dataset."""

    def __init__(self, root_dir, transform=None, seg_to_use=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Listing all the names of the images and segmentations
        if seg_to_use is None:
            seg_to_use = [True, True, False]
        self.dataset_name = 'LiTSData'
        self.root_dir = root_dir
        self.lstFilesNii = []  # create an empty list with images
        self.lstFilesNiiSeg = []  # create an empty list with segmentations

        for dirName, subdirList, fileList in os.walk(self.root_dir):
            for filename in fileList:
                if ".nii" in filename.lower():  # check whether the file's .nii
                    # check wether a file has a segmentation from a specific person
                    if "volume" in filename.lower():
                        self.lstFilesNii.append(os.path.join(dirName, filename))
                    elif "segmentation" in filename.lower():
                        self.lstFilesNiiSeg.append(os.path.join(dirName, filename))

        self.lstFilesNii.sort()
        self.lstFilesNiiSeg.sort()
        self.transform = transform
        self.seg_to_use = seg_to_use

    def _loadImage(self, path):
        image = sitk.ReadImage(path)
        return image

    def _loadSeg(self, path):
        segmentation = sitk.ReadImage(path)
        if self.seg_to_use:
            self._select_seg_to_use(segmentation, self.seg_to_use)
        return segmentation

    # TODO implement method that allows for deletion of non-essential segmentation material
    # Function that allows for specific selection of what segmentations should be used for training the model
    # (e.g. background, organ, tumor) should be implemented specifically for each dataset.
    def _select_seg_to_use(self, full_seg, seg_to_use):
        if seg_to_use == [True, True, True]:
            return full_seg
        full_seg_np = sitk.GetArrayFromImage(full_seg)
        if seg_to_use == [True, True, False]:
            # Change all tumor values (which are noted with a 2 in the seg_to_use) to 1 (i.e. liver values)
            full_seg_np[full_seg_np == 2] = 1
        elif seg_to_use == [True, False, True]:
            # Change all liver values (which are noted with a 1 in the seg_to_use) to 0 (i.e. background values)
            full_seg_np[full_seg_np == 1] = 0
        else:
            raise ValueError(seg_to_use, 'is not a valid configuration for selecting what segmentation to use. '
                                         'Having the background and one other structure to segments enabled is '
                                         'required. Change seg_to_use parameter.')

        # return the sitk image with the values changed
        return sitk.GetImageFromArray(full_seg_np)


    def __len__(self):
        return len(self.lstFilesNii)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.lstFilesNii[idx]
        seg_name = self.lstFilesNiiSeg[idx]

        image = self._loadImage(self.lstFilesNii[idx])
        segmentation = self._loadSeg(self.lstFilesNiiSeg[idx])

        sample = {'name_img': img_name, 'name_seg': seg_name, 'image': image, 'segmentation': segmentation}
        if self.transform:
            sample = self.transform(sample)

        return sample
