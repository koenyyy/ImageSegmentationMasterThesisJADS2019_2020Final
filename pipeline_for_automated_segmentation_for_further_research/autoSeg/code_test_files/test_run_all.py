import autoSeg.run.run_all as run_all


data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"
run_all.Run_all(data_dir, config_file).run()

