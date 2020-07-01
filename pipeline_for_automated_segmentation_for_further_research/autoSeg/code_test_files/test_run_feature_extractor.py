import json
import autoSeg.run.manager as manager


class RunExtractor(object):
    def __init__(self, data_dir, end_to_end_config_file_loc):
        self.data_dir = data_dir
        self.end_to_end_config_file_loc = end_to_end_config_file_loc

        # Loading the json data from specified config file
        with open(self.end_to_end_config_file_loc, 'r') as config_json:
            config_json_data = json.load(config_json)
        self.end_to_end_config_file = config_json_data

    def run(self):
        config_loc = self.end_to_end_config_file['jobs']['preprocess_unet']['configfile']
        manager.FeatureExtractor(self.data_dir, config_loc).extract_features()

if __name__ == "__main__":
    data_dir = "D:\\Thesis\\Data\\LiTS Preprocessed\\LiTS Res 1"
    config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"
    RunExtractor(data_dir, config_file).run()