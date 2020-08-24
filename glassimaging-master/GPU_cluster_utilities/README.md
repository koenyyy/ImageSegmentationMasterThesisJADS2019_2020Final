# GPU Cluster Scripts

This document aims to explain the files included in this folder. Its important to check if all file paths still hold true, these may have changed.


## Files

putty script.txt
functions for upsample_and_eval LiTS BTD ERGO implementation.txt


rename_LiTS_niis.py
train_jobs_generator.py

merge_segmentations.py
run_merge_segmentations.sh

resample_dataset.py
run_resample_dataset.sh


upsample_and_eval.py
upsample_and_eval.sh



## Usage
### Cluster script.txt
This file contains some basic commands to run code in the cluster: CD-ing to the right locations, loading a python module, mounting to the bigr cpu cluster and running experiments and jobs.

### functions for upsample_and_eval LiTS BTD ERGO implementation.txt
This file contains 3 implementations for the upsample_and_eval.py script. It contains an implementation for using with BTD, LiTS and ERGO. It's not good practice but this is what I used for running the upsampling and evaluation operations on resampled experiment results.

### rename_LiTS_niis.py
A python script that i used to rename the LiTS dataset samples to be consistent with the naming of the BTD dataset. Using the original naming convention of this dataset gave some troubles with patient ids when running experiments.

### train_jobs_generator.py
This script allows you to schedule multiple experiments on the gpu cluster at once. Depending on all the preprocessing setups that need to be tested, this script creates all the experiments automatically. Instead of calling the experiment command 24 times using different settings and namings, you can also run this script once. 

### merge_segmentations.py
This script was used to merge multiple segmentation files into one segmentation file for the ergo dataset. This dataset contained 4 individual segmentations for each of the bones in the knee. Using this script the 4 files were combined and defined as a single segmentation ground truth.

### run_merge_segmentations.sh
To run the merge_segmentations.py script this script should be called from the GPU cluster.

### resample_dataset.py
This script was used to resample a whole dataset using a resampling factor of 2 and of 4.

### resample_dataset.sh
To run the resample_dataset.py script this script should be called from the GPU cluster.


### upsample_and_eval.py
This script was used to upsample and evaluate the results of an experiment which made use of resampling factors of 2 or 4. This is needed as the results of the expriments using a resampling factor do not have 

### upsample_and_eval.sh
To run the upsample_and_eval.py script this script should be called from the GPU cluster.