import autoSeg.run.run_all as run_all
import os
import sys


class RunExperiment(object):
    def __init__(self, data_dir, config_dir):
        self.data_dir = data_dir
        self.config_dir = config_dir

    def run(self):
        print("Running experiment")
        print("Checking existing directories")
        self.make_data_dir()

        runner = run_all.Run_all(self.data_dir, self.config_dir)
        runner.run()


    def make_data_dir(self):
        # check if data dir already exists, if not create dir
        data_path = os.path.join(sys.path[0], 'data')
        if os.path.isdir(data_path):
            if not os.path.isdir(os.path.join(data_path, 'input_data')):
                os.makedirs(os.path.join(data_path, 'input_data'))
            if not os.path.isdir(os.path.join(data_path, 'intermediate_data')):
                os.makedirs(os.path.join(data_path, 'intermediate_data'))
            if not os.path.isdir(os.path.join(data_path, 'preprocessed_data')):
                os.makedirs(os.path.join(data_path, 'preprocessed_data'))
        else:
            os.makedirs(data_path)
            os.makedirs(os.path.join(data_path, 'input_data'))
            os.makedirs(os.path.join(data_path, 'intermediate_data'))
            os.makedirs(os.path.join(data_path, 'preprocessed_data'))

data_dir = os.path.join(sys.path[0],'data','input_data')
config_file = os.path.join(sys.path[0], 'autoSeg', 'config', 'preprocess_data.json')
rep = RunExperiment(data_dir, config_file)
rep.run()