import importlib

import torch
import torchvision.transforms as transforms
import json
import os
import sys
from autoSeg.models.UNet3D import UNet3D
import autoSeg.utils.utils as ud
from autoSeg.preprocessing.Transform import OtsuCrop, ResampleVxSpacing, N4BiasCorrection, Normalize, IntensityClipper, ToTensor
import pandas as pd
import plotly.graph_objects as go


class ModelEvaluator(object):
    def __init__(self, INPUT_data_dir, eval_config_loc, setup_config_loc):
        self.INPUT_data_dir = INPUT_data_dir

        # Loading the json data from specified config file
        eval_config_loc = os.path.join(sys.path[0], 'autoSeg', 'config', eval_config_loc)
        with open(eval_config_loc, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.config_file = config_json_data

        # Loading the json data from specified config file
        setup_config_loc = os.path.join(sys.path[0], 'autoSeg', 'config', setup_config_loc)
        with open(setup_config_loc, 'r') as setup_config_json:
            setup_config_json_data = json.load(setup_config_json)
        self.setup_config_file = setup_config_json_data

        self.dataset_import = getattr(
            importlib.import_module('autoSeg.data_loading.{0:s}'.format(self.config_file['dataset'][0])),
            '{0:s}'.format(self.config_file['dataset'][1]))
        # exec('from autoSeg.data_loading.{0:s} import {0:s}'.format(self.config_file['dataset'][0],
        #                                                            self.config_file['dataset'][1]))

        self.prepped_data = os.path.join(sys.path[0], 'data', 'preprocessed_data')
        self.dataset = self.dataset_import(root_dir=self.prepped_data,
                                           transform=transforms.Compose([ToTensor()]), seg_to_use=self.config_file['seg_to_use'])

        self.dataloader = torch.utils.data.DataLoader(self.dataset)

    def evaluate_model(self):
        # Load the trained model
        net = self.load_saved_model()

        if self.config_file.get('outputtype') == 'show':
            # show batch of data
            self.show_segmentations(net)
            self.create_train_plot(outputtype='show')
        elif self.config_file.get('outputtype') == 'save':
            # save batch
            self.save_segmentations(net)
            self.create_train_plot(outputtype='save')



    def load_saved_model(self):
        unet_params = self.setup_config_file['Model'][1]
        net = UNet3D(**unet_params).float()

        # getting to the right folder and trained network
        experiment_results_loc = os.path.join(sys.path[0], 'experiment_results')
        experiment_results_loc = next(os.walk(experiment_results_loc))
        latest_trained_net_loc = os.path.join(os.path.join(experiment_results_loc[0], experiment_results_loc[1][-1]),
                                      'trained_3DUnet.pth')


        # load state dictionary of the network
        net.load_state_dict(torch.load(latest_trained_net_loc))
        return net

    def show_segmentations(self, net):
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader):
                print(i_batch, sample_batched['image'].size(),
                      sample_batched['segmentation'].size())

                output = net(sample_batched['image'].float())
                # observe batch.
                # displaying 20th slice
                ud.showImageBatch({'image': sample_batched['image'], 'segmentation': output, 'ground_truth': sample_batched['segmentation']}, 20)

                # in case we only need to show first
                if self.config_file.get('onlyfirst'):
                    break

    def save_segmentations(self, net):
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader):
                print(i_batch, sample_batched['image'].size(),
                      sample_batched['segmentation'].size())

                output = net(sample_batched['image'].float())

                # observe batch.
                # displaying 20th slice
                ud.saveImageBatch({'image': sample_batched['image'], 'segmentation': output,'ground_truth': sample_batched['segmentation']}, 20, batchNr=i_batch)

                # in case we only need to show first
                if self.config_file.get('onlyfirst'):
                    break

    def create_train_plot(self, outputtype):
        # getting to the right folder and trained network
        experiment_results_loc = os.path.join(sys.path[0], 'experiment_results')
        experiment_results_loc = next(os.walk(experiment_results_loc))
        results_loc = os.path.join(os.path.join(experiment_results_loc[0], experiment_results_loc[1][-1]),
                                      'results.csv')

        df = pd.read_csv(results_loc)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss_score'],
                                 mode='lines+markers',
                                 name='Train Loss'))

        fig.add_trace(go.Scatter(x=df['epoch'], y=df['validation_loss_score'],
                                 mode='lines+markers',
                                 name='Validation Loss'))

        # TODO add train evaluation
        # fig.add_trace(go.Scatter(x=random_x, y=random_y1,
        #                          mode='lines+markers',
        #                          name='Test'))

        if outputtype == 'save':
            # getting to the right folder and trained network
            experiment_results_loc = os.path.join(sys.path[0], 'experiment_results')
            experiment_results_loc = next(os.walk(experiment_results_loc))
            train_plot_loc = os.path.join(os.path.join(experiment_results_loc[0], experiment_results_loc[1][-1]),
                                          'train_plot.html')
            if self.config_file.get('onlyfirst'):
                # in case only the first image is segmented we are probably debugging.
                # Therefore we also display the loss curve
                fig.write_html(train_plot_loc, auto_open=True)
            else:
                fig.write_html(train_plot_loc, auto_open=False)
        else:
            fig.show()
