import platform
import glob
import os
import sys

import pandas as pd
from glassimaging.dataloading.niftidataset import NiftiDataset
import logging
import json
from torch.utils.data import Dataset
import numpy as np

class ErgoData(NiftiDataset):
    available_sequences = ['t1t2']

    """The image paths for each subject are stored at initialization
    """

    def __init__(self, df=None):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df

    def importData(self, data_loc, nsplits=4):
        # Listing all the names of the images and segmentations
        lstFilesGz = []  # create an empty list with images
        lstFilesGzSeg = []  # create an empty list with segmentations of bones

        for dirName, subdirList, fileList in os.walk(data_loc):
            for filename in fileList:
                if ".nii" in filename.lower():  # check whether the file's .nii
                    # check whether a file has a segmentation from a specific person
                    if "T1_T2_Morphology" in filename.lower():
                        lstFilesGz.append(os.path.join(dirName, filename))
                    if "combined_bone_mask" in filename.lower():
                        lstFilesGzSeg.append(os.path.join(dirName, filename))

        lstFilesGz.sort()
        lstFilesGzSeg.sort()


        images = {"{:03d}".format(x+1): i for x, i in enumerate(lstFilesGz)}
        segmentation = {"{:03d}".format(y+1): j for y, j in enumerate(lstFilesGzSeg)}

        patients = images.keys()
        df = pd.DataFrame.from_dict(images, orient='index', columns=['t1t2'])

        for p in patients:
            df.at[p, 'seg'] = segmentation[p]

        self.df = df
        self.patients = patients

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """

    @staticmethod
    def fromFile(loc, nsplits=4):
        instance = ErgoData()
        instance.importData(loc, nsplits)
        logging.info('ErgoData new dataloader created from ' + loc + '.')
        return instance

    def setSplits(self, splits_file):
        """ Load the information on cross-validation splits from a json file
        """
        with open(splits_file, 'r') as file:
            splits = json.load(file)
        # Set all patient to split -1, so that only patients in the actual splits file are included
        self.df['split'] = -1
        for i in range(0, len(splits)):
            for p in splits[i]:
                self.df.at[p, 'split'] = i

    def getDataset(self, splits=(), sequences=None, transform=None, preprocess_config=None):
        if len(splits) == 0:
            splits = range(0, self.nsplits)
        if sequences is None:
            sequences = self.available_sequences
        dataset = ErgoDataset(self.df.loc[[s in splits for s in self.df['split']]], sequences,
                                     transform=transform, preprocess_config=preprocess_config)
        return dataset


class ErgoDataset(NiftiDataset, Dataset):

    def __init__(self, dataframe, sequences, transform=None, preprocess_config=None):
        Dataset.__init__(self)
        NiftiDataset.__init__(self)
        self.df = dataframe
        self.sequences = sequences
        self.patients = self.df.index.values
        self.transform = transform
        self.config_file = preprocess_config

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patientname = self.patients[idx]
        (image, segmentation) = self.loadSubjectImages(patientname, self.sequences,
                                                       normalized=self.config_file['use_normalization'],
                                                       technique=self.config_file['technique'],
                                                       using_otsu_ROI=self.config_file['using_otsu_ROI'],
                                                       resampling_factor=self.config_file['resampling_factor'])

        # normal segmentations in Ergo dataset contains distinct bones, here we're only interested in the collection of bones
        segmentation[segmentation == 2] = 1
        segmentation[segmentation == 3] = 1
        segmentation[segmentation == 4] = 1

        seg_file = self.getFileName(patientname, 'seg')
        sample = {'data': image, 'seg': segmentation.astype(np.uint8), 'seg_file': seg_file, 'subject': patientname}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def saveListOfPatients(self, path):
        with open(path, 'w') as file:
            json.dump(self.patients.tolist(), file)
