import datetime
import json
import os
import sys

from autoSeg.preprocessing.feature_extractor import FeatureExtractorObject
from autoSeg.preprocessing.initial_preprocessing import InitialPreprocessor
from autoSeg.model_training.train_model import ModelTrainer
from autoSeg.model_evaluation.evaluate_model import ModelEvaluator


class FeatureExtractor(object):
    """
    Object for doing feature extraction of a given dataset
    """
    def __init__(self, data_dir, preprocess_config_loc):
        self.data_dir = data_dir
        self.preprocess_config_loc = preprocess_config_loc

    def extract_features(self):
        print("DOING FEATURE EXTRACTION:")
        # Run feature extraction on dataset of medical images
        FeatureExtractorObject(self.data_dir, self.preprocess_config_loc).run()

class Preprocessor(object):
    """
    Object for doing preprocessing on a given dataset
    """
    def __init__(self, data_dir, preprocess_config_loc):
        self.data_dir = data_dir
        self.preprocess_config_loc = preprocess_config_loc

    def preprocess(self):
        print("PREPROCESSING:")

        # Run first initial processing to crop, correct bias and resample
        InitialPreprocessor(self.data_dir, self.preprocess_config_loc).run()



class Trainer(object):
    """
    Object for doing training on a given dataset
    """
    def __init__(self, setup_config_loc, train_config_loc):
        self.setup_config_loc = setup_config_loc
        self.train_config_loc = train_config_loc

        # Loading the json data from specified config file
        self.setup_config_loc = os.path.join(sys.path[0], 'autoSeg', 'config', self.setup_config_loc)
        self.train_config_loc = os.path.join(sys.path[0], 'autoSeg', 'config', self.train_config_loc)
        with open(self.train_config_loc, 'r') as train_config:
            train_config_json_data = json.load(train_config)
        self.train_config = train_config_json_data

    def train(self):
        print("TRAINING:")
        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        experiment_results_location = os.path.join(sys.path[0], 'experiment_results', str(now) + '_' + str(self.train_config['dataset'][0]))

        ModelTrainer(self.setup_config_loc, self.train_config_loc).train_model(experiment_results_location)




class Evaluator(object):
    """
    Object for doing evaluation on a given dataset
    """

    def __init__(self, data_dir, eval_config_loc, setup_config_loc):
        self.data_dir = data_dir
        self.eval_config_loc = eval_config_loc
        self.setup_config_loc = setup_config_loc

    def evaluate(self):
        print("EVALUAING:")
        ModelEvaluator(self.data_dir, self.eval_config_loc, self.setup_config_loc).evaluate_model()




class Tester(object):
    """
    Object for doing testing on a given dataset
    """
    def __init__(self, test_config):
        self.test_config = test_config

    def test(self):
        print("Testing:")
