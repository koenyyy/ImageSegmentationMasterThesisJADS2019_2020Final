# -*- coding: utf-8 -*-
"""
Job to evaluate a network

@author: Karin van Garderen
"""
from glassimaging.execution.jobs.job import Job
import sys
import os
import argparse
from torch.utils.data import DataLoader
from glassimaging.evaluation.evaluator import StandardEvaluator
from glassimaging.evaluation.utils import plotResultImageWithoutGT
from glassimaging.dataloading.brats18 import Brats18
from glassimaging.dataloading.btd import BTD
from glassimaging.dataloading.generic import GenericData
from glassimaging.dataloading.transforms.totensor import ToTensor

from glassimaging.preprocessing.set_patch_size import set_patch_size


class JobApply(Job):

    def __init__(self, configfile, name, tmpdir, homedir=None, uid=None):
        super().__init__(configfile, name, tmpdir, homedir, uid=uid)

    def getSchema(self):
        return {
            "type": "object",
            "properties": {
                "Nifti Source": {"type": "string"},
                "Splits": {"type": "array"},
                "Model Source": {"type": "string"},
                "Patch size": {"type": "array"},
                "Batch size": {"type": "integer"},
                "Output type": {"type": "string"},
                "Dataset": {"type": "string"},
                "Only first": {"type": "boolean"},
                "Brainmask": {"type": "boolean"},
            },
            "required": ["Nifti Source", "Patch size", "Batch size", "Dataset"]
        }

    def run(self):
        ##### Create identifiers
        myconfig = self.config

        ##### Set network specifics
        # check if patch size isn't too large. If so make it smaller
        # patchsize = myconfig["Patch size"]
        patchsize = tuple(set_patch_size(myconfig))
        batchsize = myconfig["Batch size"]
        output_type = myconfig["Output type"]
        only_first = myconfig["Only first"]

        #### Data specifics
        splits = myconfig["Splits"]

        ##### load datamanager
        loc = myconfig["Nifti Source"]
        if self.config['Dataset'] == 'Brats18':
            dataset = Brats18.fromFile(loc)
        elif self.config['Dataset'] == 'BTD':
            if self.config['Brainmask'] == False:
                dataset = BTD.fromFile(loc, brainmask=False)
            else:
                dataset = BTD.fromFile(loc, brainmask=True)
        elif self.config['Dataset'] == 'Generic':
            dataset = GenericData.fromFile(loc)

        if "Splits from File" in myconfig:
            dataset.loadSplits(myconfig["Splits from File"])

        ##### Load model from source step
        sourcestep = myconfig["Model Source"]
        loc_model = os.path.join(self.datadir, sourcestep)
        config_model = self.loadConfig(os.path.join(loc_model, 'config.json'))
        sequences = config_model["Sequences"]

        transform = ToTensor()
        testset = dataset.getDataset(splits, sequences, transform=transform, preprocess_config=self.config)
        dataloader = DataLoader(testset, batch_size=batchsize, num_workers=0, shuffle=True)

        evaluator = StandardEvaluator.loadFromCheckpoint(os.path.join(loc_model, 'model.pt'))
        self.logger.info('Dataloader has {n} images.'.format(n=len(testset)))

        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched["data"]
            subjects = sample_batched["subject"]
            header_sources = sample_batched["header_source"]
            t1_sources = sample_batched["t1_source"]
            resultpaths = [os.path.join(self.tmpdir, s + '_segmented.nii.gz') for s in subjects]
            # check if patch size isn't too large. If so make it smaller
            print("jobeval", type(images), print(images))
            evaluator.segmentNifti(images, header_sources, patchsize, resultpaths)
            for i in range(0, len(subjects)):
                plotResultImageWithoutGT(t1_sources[i], resultpaths[i], self.tmpdir,
                                            subjects[i], output_type=output_type)
                self.logger.info('Nifti image segmented for {}.'.format(subjects[i]))
            if only_first: break
        self.tearDown()

def main(configfile, name, tmpdir, homedir=None):
    job = JobApply(configfile, name, tmpdir, homedir)
    job.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a job.')
    parser.add_argument('name', help='a name to call your job by.')
    parser.add_argument('configfile', help='path to a json config file.')
    parser.add_argument('tmpdir', help='directory for the output.')
    parser.add_argument('--log', help='additional directory to write logs to.')
    args = parser.parse_args()
    main(args.configfile, args.name, args.tmpdir, args.log)
