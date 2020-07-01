import importlib
import os
import sys

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import numpy as np

from collections import OrderedDict
from collections import namedtuple
from itertools import product
from IPython import display
import time

from autoSeg.preprocessing.Transform import OtsuCrop, ResampleVxSpacing, N4BiasCorrection, Normalize, IntensityClipper, ToTensor
import autoSeg.utils.utils as ud
from autoSeg.models.diceloss import DiceLoss
from autoSeg.models.UNet3D import UNet3D


class RunBuilder():
    @staticmethod
    def get_runs(params):
        # TODO make random instead of taking all permutations
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.validation_loss = 0
        self.num_correct = 0
        self.start_time = None


class Run():
    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None


class RunManager():
    def __init__(self):
        self.epoch = Epoch()

        self.run = Run()

        self.network = None
        self.loader = None

    def begin_run(self, run_params, network, loader):
        self.run.start_time = time.time()

        self.run.params = run_params
        self.run.count += 1

        self.network = network
        self.loader = loader

        # images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)

    def end_run(self):
        self.epoch.count = 0

    def begin_epoch(self):
        self.epoch.start_time = time.time()
        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.validation_loss = 0
        self.epoch.num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        loss = self.epoch.loss / len(self.loader.dataset)
        validation_loss = self.epoch.validation_loss / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)

        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss_score"] = loss
        results["validation_loss_score"] = validation_loss
        results["accuracy_score"] = accuracy
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration

        for k, v in self.run.params._asdict().items():
            results[k] = v

        self.run.data.append(results)
        df = pd.DataFrame.from_dict(self.run.data, orient='columns')

        display.clear_output(wait=True)
        display.display(df)

    def track_loss(self, loss):
        self.epoch.loss += loss.item() * self.loader.batch_size

    def track_validation_loss(self, validation_loss):
        self.epoch.validation_loss += validation_loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch.num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        fileName = os.path.join(fileName, 'results')
        pd.DataFrame.from_dict(
            self.run.data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)

########################################################################################################################
################################## Below we start the training process #################################################
########################################################################################################################


class ModelTrainer(object):
    def __init__(self, setup_config_loc, train_config_loc):
        # Loading the json data from specified config file
        with open(setup_config_loc, 'r') as setup_config_json:
            setup_config_json_data = json.load(setup_config_json)
        self.setup_config_file = setup_config_json_data

        # Loading the json data from specified config file
        with open(train_config_loc, 'r') as train_config_json:
            train_config_json_data = json.load(train_config_json)
        self.train_config_file = train_config_json_data

        # Loading the right dataset based on the config file
        # TODO explain the setting of the datastet file [0] and the dataset class [1]
        self.dataset_import = getattr(importlib.import_module('autoSeg.data_loading.{0:s}'.format(self.train_config_file['dataset'][0])),
                                      '{0:s}'.format(self.train_config_file['dataset'][1]))

        self.prepped_data = os.path.join(sys.path[0],'data', 'preprocessed_data')


        #Check what dataset is needed
        # self.dataset = eval(self.train_config_file['dataset'][1])(root_dir=self.prepped_data,
        #                               transform=transforms.Compose([ToTensor()]))
        self.dataset = self.dataset_import(root_dir=self.prepped_data, transform=transforms.Compose([ToTensor()]), seg_to_use=self.train_config_file['seg_to_use'])

        # TODO test set to do final segmentation on (can also be done on validation set)
        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        val_split = int(math.floor(self.train_config_file['validationsetpct'] * dataset_size))
        test_split = int(math.floor(self.train_config_file['testsetpct'] * dataset_size))
        random_seed = 13
        # shuffle indices to get random train validation split
        if self.train_config_file['shuffle']:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[val_split + test_split:], indices[:val_split], indices[val_split:val_split + test_split]
        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.validation_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(val_indices)

        self.dataloader = DataLoader(self.dataset)


    def display_batch(self):
        for i_batch, sample_batched in enumerate(self.dataloader):
            print(i_batch, sample_batched['image'].size(),
                  sample_batched['segmentation'].size())

            # observe 4th batch and stop.
            if i_batch == 1:
                plt.figure()
                ud.showImageBatch(sample_batched)
                plt.axis('off')
                plt.ioff()
                plt.show()
                break

    def get_loss(self, loss, weights):
        loss_function = None

        if loss.lower() == 'dice':
            loss_function = DiceLoss(weights)
        elif loss.lower() == 'crossentropy':
            pass
        else:
            # use dice loss as a default
            loss_function = DiceLoss(weights)
        return loss_function

    def train_model(self, experiment_results_location):
        m = RunManager()
        for run_params in RunBuilder.get_runs(self.train_config_file['variableparams']):
            # get the unet paramters form the setup_unet config file
            unet_params = self.setup_config_file['Model'][1]
            network = UNet3D(**unet_params).float()


            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device) # send network to GPU device if available
            print('running on:', device)

            # train_loader = torch.utils.data.DataLoader(self.training_set, shuffle=run_params.shuffle,  batch_size=run_params.batchsize,
            #                                            num_workers=run_params.numworkers)

            train_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=run_params.batchsize,
                                                       num_workers=run_params.numworkers,
                                                       sampler=self.train_sampler)

            validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                            batch_size=run_params.batchsize,
                                                            num_workers=run_params.numworkers,
                                                            sampler=self.validation_sampler)

            # TODO check if variable learning rates over time are possible (different lr in different epochs)
            #  (not necessary for Adam as this is already adaptive and uses momentum )
            optimizer = optim.Adam(network.parameters(), lr=run_params.learningrate)

            ###########Possibly add TensorBoard code here###########
            ###########                                  ###########
            ########################################################

            m.begin_run(run_params, network, train_loader)
            for epoch in range(run_params.epochs):
                m.begin_epoch()
                for batch in train_loader:
                    # images_batch, segmentations_batch = \
                    #     batch['image'], batch['segmentation'] # get batch
                    images_batch, segmentations_batch = \
                        batch['image'].to(device), batch['segmentation'].to(device) # get batch
                    preds = network(images_batch.float())  # Pass Batch
                    # print(preds.size(), batch['segmentation'].size())
                    # print(preds.requires_grad, batch['segmentation'].requires_grad)

                    ############
                    # print("####")
                    # # print(torch.min(preds[0][0]), torch.max(preds[0][0]), torch.min(preds[0][1]), torch.max(preds[0][1]))
                    # print(torch.min(images_batch), torch.max(images_batch), torch.min(segmentations_batch), torch.max(segmentations_batch))
                    # print("####")

                    # ground_truth = torch.cat((images_batch.float(), segmentations_batch.float()), 1)
                    ground_truth = segmentations_batch

                    loss = run_params.loss
                    weights = run_params.lossweights
                    criterion = self.get_loss(loss, weights)

                    # print("####")
                    # print(preds.size(), ground_truth.size())
                    # print(torch.min(preds), torch.max(preds), torch.min(ground_truth), torch.max(ground_truth))
                    # print("####")

                    loss = criterion(preds, ground_truth)

                    optimizer.zero_grad()  # Zero the Gradients
                    loss.backward()  # Calculate the Gradients
                    optimizer.step()  # Update the Weights

                    m.track_loss(loss)
                #                 m.track_num_correct(preds, segmentations)
                # vgm moet hier de validation komen
                for batch in validation_loader:
                    images_batch, segmentations_batch = \
                        batch['image'].to(device), batch['segmentation'].to(device)  # get batch
                    preds = network(images_batch.float())  # Pass Batch

                    ############
                    # ground_truth = torch.cat((images_batch.float(), segmentations_batch.float()), 1)
                    ground_truth = segmentations_batch

                    validation_loss = run_params.loss
                    weights = run_params.lossweights
                    criterion = self.get_loss(validation_loss, weights)

                    validation_loss = criterion(preds, ground_truth)
                    print(validation_loss)
                    ###########
                    m.track_validation_loss(validation_loss)
                m.end_epoch()
            m.end_run()
            # Check if folder for saving exists, else we create the folder and store the model there
            if not os.path.isdir(experiment_results_location):
                os.makedirs(experiment_results_location)
                # also create an image folder for storing images later on
                image_folder = os.path.join(experiment_results_location, 'images')
                os.makedirs(image_folder)
        m.save(experiment_results_location)

        # Also save the model itself to the experiment_results folder
        save_path = os.path.join(experiment_results_location, 'trained_3DUnet.pth')
        torch.save(network.state_dict(), save_path)