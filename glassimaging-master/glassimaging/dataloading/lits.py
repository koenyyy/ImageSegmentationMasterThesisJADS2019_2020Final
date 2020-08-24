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

class LitsData(NiftiDataset):
    available_sequences = ['ct']

    """The image paths for each subject are stored at initialization
    """

    def __init__(self, df=None):
        NiftiDataset.__init__(self)
        if df is not None:
            self.df = df

    def importData(self, data_loc, nsplits=4):
        # Listing all the names of the images and segmentations
        lstFilesGz = []  # create an empty list with images
        lstFilesGzSeg = []  # create an empty list with segmentations

        for dirName, subdirList, fileList in os.walk(data_loc):
            for filename in fileList:
                if ".nii" in filename.lower():  # check whether the file's .nii
                    # check whether a file has a segmentation from a specific person
                    if "volume" in filename.lower():
                        lstFilesGz.append(os.path.join(dirName, filename))
                    if "segmentation" in filename.lower():
                        lstFilesGzSeg.append(os.path.join(dirName, filename))

        lstFilesGz.sort()
        lstFilesGzSeg.sort()


        images = {"{:03d}".format(x+1): i for x, i in enumerate(lstFilesGz)}
        segmentations = {"{:03d}".format(y+1): j for y, j in enumerate(lstFilesGzSeg)}

        patients = images.keys()
        df = pd.DataFrame.from_dict(images, orient='index', columns=['ct'])

        for p in patients:
            df.at[p, 'seg'] = segmentations[p]

        self.df = df
        self.patients = patients

        self.createCVSplits(nsplits)

    """Create a datamanager object from the filesystem
    """

    @staticmethod
    def fromFile(loc, nsplits=4):
        instance = LitsData()
        instance.importData(loc, nsplits)
        logging.info('LitsData new dataloader created from ' + loc + '.')
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
        dataset = LitsDataset(self.df.loc[[s in splits for s in self.df['split']]], sequences,
                                     transform=transform, preprocess_config=preprocess_config)
        return dataset


class LitsDataset(NiftiDataset, Dataset):

    def __init__(self, dataframe, sequences, transform=None, preprocess_config=None):
        Dataset.__init__(self)
        NiftiDataset.__init__(self)
        self.df = dataframe
        self.sequences = sequences
        self.patients = self.df.index.values
        print(self.df['seg'][0])
        print(self.df['seg'][1])
        print(self.df['seg'][2])
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

        # normal segmentations in LiTS dataset contain tumors, we're only interested in the liver so we omit the tumors
        # we set the tumor values (2) to be of type liver (1)
        segmentation[segmentation == 2] = 1

        seg_file = self.getFileName(patientname, 'seg')
        sample = {'data': image, 'seg': segmentation.astype(np.uint8), 'seg_file': seg_file, 'subject': patientname}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def saveListOfPatients(self, path):
        with open(path, 'w') as file:
            json.dump(self.patients.tolist(), file)
