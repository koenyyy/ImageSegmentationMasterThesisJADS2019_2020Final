import json
import autoSeg.run.manager as manager

class Run_all(object):
    def __init__(self, data_dir, end_to_end_config_file_loc):
        self.data_dir = data_dir
        self.end_to_end_config_file_loc = end_to_end_config_file_loc

        # Loading the json data from specified config file
        with open(self.end_to_end_config_file_loc, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.end_to_end_config_file = config_json_data

    # TODO add try except statements and make robust (if time allows make unit tests)
    def run(self):
        eval_config_loc = self.end_to_end_config_file['jobs']['eval_nifti']['configfile']
        setup_config_loc = self.end_to_end_config_file['jobs']['setup_unet']['configfile']
        manager.Evaluator(self.data_dir, eval_config_loc, setup_config_loc).evaluate()

data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"
Run_all(data_dir, config_file).run()