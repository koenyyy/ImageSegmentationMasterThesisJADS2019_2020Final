from autoSeg.data_loading.LiTSDataset import LiTSDataset
import os

data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"

ds = LiTSDataset(data_dir, transform=None, seg_to_use=[True, False, True])

for item in ds:
    print(0)
    # print(item['name_img'])
    # print(item['name_seg'])
    # print(item['segmentation'])
    # print(item['name_img'].split(os.path.sep)[-1])
    # print(item['name_seg'].split(os.path.sep)[-1])