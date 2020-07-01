import json

import autoSeg.run.manager as manager

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Run_all(object):
    def __init__(self, data_dir, end_to_end_config_file_loc):
        self.data_dir = data_dir
        self.end_to_end_config_file_loc = end_to_end_config_file_loc

        # Loading the json data from specified config file
        with open(self.end_to_end_config_file_loc, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.end_to_end_config_file = config_json_data

    # TODO n-1. Add try except statements and make robust (if time allows make unit tests)
    def run(self):
        # TODO enable multiple modalities as input

        # TODO automatically create slurm script

        # TODO 4. create pre-process param selector (base choices on random 10 samples from ds with 35 epochs with highest avg loss decrease)

        # TODO check if test class is really needed after train and eval
        preprocess_config_loc = self.end_to_end_config_file['jobs']['preprocess_unet']['configfile']
        setup_config_loc = self.end_to_end_config_file['jobs']['setup_unet']['configfile']
        train_config_loc = self.end_to_end_config_file['jobs']['train_unet']['configfile']
        eval_config_loc = self.end_to_end_config_file['jobs']['eval_nifti']['configfile']
        test_config_loc = self.end_to_end_config_file['jobs']['eval_nifti']['configfile']

        manager.FeatureExtractor(self.data_dir, preprocess_config_loc).extract_features()

        manager.Preprocessor(self.data_dir, preprocess_config_loc).preprocess()

        # TODO n. implement patch based training
        manager.Trainer(setup_config_loc, train_config_loc).train()

        manager.Evaluator(self.data_dir, eval_config_loc, setup_config_loc).evaluate()

        manager.Tester.test(test_config_loc)
