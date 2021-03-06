# -*- coding: utf-8 -*-
"""
Job to train a network

@author: Karin van Garderen
"""

import os
import argparse
import sys
import json

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from glassimaging.execution.jobs.job import Job
from glassimaging.training.standardTrainer import StandardTrainer
from glassimaging.dataloading.brats18 import Brats18
from glassimaging.dataloading.btd import BTD
from glassimaging.dataloading.hippocampus import Hippocampus
from glassimaging.dataloading.lipodata import LipoData
from glassimaging.dataloading.lits import LitsData
from glassimaging.dataloading.ergo import ErgoData
from glassimaging.dataloading.transforms.randomcrop import RandomCrop
from glassimaging.dataloading.transforms.totensor import ToTensor
from glassimaging.dataloading.transforms.compose import Compose
from glassimaging.dataloading.transforms.binaryseg import BinarySegmentation
from glassimaging.models.diceloss import DiceLoss

from glassimaging.evaluation.utils import logDataLoader

from glassimaging.preprocessing.set_patch_size import set_patch_size

from datetime import datetime

class JobTrain(Job):

    def __init__(self, configfile, name,  tmpdir, homedir = None, uid = None):
        super().__init__(configfile, name,  tmpdir, homedir, uid = uid)

    def getSchema(self):
        return {
            "type": "object",
            "properties": {
                "Nifti Source": {"type": "string"},
                "Splits": {"type": "array"},
                "Model Source": {"type": "string"},
                "Patch size": {"type": "array"},
                "Batch size": {"type": "integer"},
                "Dataset": {"type": "string"},
                "Epochs": {"type": "integer"},
                "Num Workers": {"type": "integer"},
                "Loss": {"type": "string",
                         "enum": ["crossentropy", "dice"]},
                "Testsplits": {"type": "array"},
                "Target": {"type": "string",
                         "enum": ["Brainmask"]},
                "Whole Tumor": {"type": "boolean"},
                "Splits from File": {"type": "string"},
                "Sequences": {"type": "array"},
                "Brainmask": {"type": "boolean"},
            },
            "required": ["Nifti Source", "Patch size", "Batch size", "Dataset", "Epochs", "Num Workers",
                         "Testsplits", "Splits"]
        }

    def getDataloader(self):
        batchsize = self.config['Batch size']
        loc = os.path.join(self.datadir, self.config["Nifti Source"])
        sequences = self.config["Sequences"] if "Sequences" in self.config else None
        # Loading the specialized config settings from file
        print("normalization technique is:", self.config["technique"])
        print("using otsu threshold for normalization:", self.config["using_otsu_ROI"])
        print("resampling factor is:", self.config["resampling_factor"])

        if self.config['Dataset'] == 'Brats18':
            dataset = Brats18.fromFile(loc)
        elif self.config['Dataset'] == 'BTD':
            print(self.config['Brainmask'])
            if self.config['Brainmask'] == False:
                dataset = BTD.fromFile(loc, brainmask=False)
            else:
                dataset = BTD.fromFile(loc, brainmask=True)
        elif self.config['Dataset'] == 'Hippocampus':
            dataset = Hippocampus.fromFile(loc)
        elif self.config['Dataset'] == 'LipoData':
            dataset = LipoData.fromFile(loc)
        elif self.config['Dataset'] == 'LitsData':
            dataset = LitsData.fromFile(loc)
        elif self.config['Dataset'] == 'ErgoData':
            dataset = ErgoData.fromFile(loc)

        if "Splits from File" in self.config:
            dataset.setSplits(self.config["Splits from File"])
        elif "Crossval Splits" in self.config:
            dataset.createCVSplits(self.config["Crossval Splits"])
        #### Data specifics
        splits = self.config['Splits']
        testsplits = self.config['Testsplits']
        dataset.saveSplits(self.tmpdir)
        # check if patch size isn't too large. If so make it smaller
        # targetsize = tuple(self.config["Patch size"])
        targetsize = tuple(set_patch_size(self.config))

        imgsize = targetsize

        transforms = [
            RandomCrop(output_size=imgsize),
            ToTensor()
        ]
        if 'Whole Tumor' in self.config and self.config["Whole Tumor"]:
            transforms = [BinarySegmentation()] + transforms
        transform = Compose(transforms)

        if 'Target' in self.config and self.config['Target'] == 'Brainmask':
            trainset = dataset.getBrainmaskDataset(splits, sequences, transform=transform)
        else:
            trainset = dataset.getDataset(splits, sequences, transform=transform, preprocess_config=self.config)

        trainset.saveListOfPatients(os.path.join(self.tmpdir, 'trainset.json'))
        self.logger.info('Generating patches with input size ' + str(imgsize) + ' and outputsize ' + str(targetsize))

        n_workers = self.config['Num Workers']
        now = datetime.now()
        print('train loader is initializing', now)
        trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=n_workers, shuffle=True)
        later = datetime.now()
        print('train loader is done initializing', later-now)
        if len(testsplits) > 0:
            if 'Target' in self.config and self.config['Target'] == 'Brainmask':
                testset = dataset.getBrainmaskDataset(testsplits, sequences, transform=transform)
            else:
                testset = dataset.getDataset(testsplits, sequences, transform=transform, preprocess_config=self.config)
            testloader = DataLoader(testset, batch_size=batchsize, num_workers=n_workers, shuffle=True)
        else:
            testloader = None

        logDataLoader(trainloader, self.tmpdir)

        return trainloader, testloader

    def run(self):
        myconfig = self.config

        epochs = myconfig['Epochs']
        maxBatchesPerEpoch = myconfig["Batches per epoch"]

        ##### Load model from source step
        sourcestep = myconfig["Model Source"]
        model_loc = os.path.join(self.datadir, sourcestep)
        trainer = StandardTrainer.loadFromCheckpoint(os.path.join(model_loc, 'model.pt'))
        trainer.setLogger(self.logger)
        model_desc = trainer.getModelDesc()
        model = model_desc[0]
        self.logger.info('model loaded from ' + model_loc + '.')

        ### set loss function
        if 'Loss' in myconfig:
            if myconfig['Loss'] == 'dice':
                if 'loss_weights' in myconfig:
                    trainer.setLossFunction(DiceLoss(weights=tuple(myconfig['loss_weights'])))
                else:
                    trainer.setLossFunction(DiceLoss())
            elif myconfig['Loss'] == 'crossentropy':
                trainer.setLossFunction(CrossEntropyLoss())

        (trainloader, testloader) = self.getDataloader()

        trainer.trainWithLoader(trainloader, epochs, testloader=testloader, maxBatchesPerEpoch=maxBatchesPerEpoch)
        self.logger.info('training finished for ' + str(epochs) + ' epochs.')
        trainer.writeLog(self.tmpdir)
        modelpath = os.path.join(self.tmpdir, 'model.pt')
        trainer.saveModel(modelpath)
        self.logger.info('model and training log saved under ' + modelpath + '.')
        del trainer
        self.tearDown()



def main(configfile, name, tmpdir, homedir=None):
    job = JobTrain(configfile, name, tmpdir, homedir)
    job.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a job.')
    parser.add_argument('name', help='a name to call your job by.')
    parser.add_argument('configfile', help='path to a json config file.')
    parser.add_argument('tmpdir', help='directory for the output.')
    parser.add_argument('--log', help='additional directory to write logs to.')
    args = parser.parse_args()
    main(args.configfile, args.name, args.tmpdir, args.log)
