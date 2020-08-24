# BIGR CPU Cluster Scripts

This document aims to explain the files included in this folder. Its important to check if all file paths still hold true, these may have changed. Furthermore, these scripts should be located on the BIGR cluster and be executed there.

## Files

bigr-app001 call n4.txt
make_n4bc_array_script.py
copy_segmentations.sh
create_subtree.sh
n4bc_array.sh

## Usage
### bigr-app001 call n4.txt
This document tells how to perform the n4 bias correction and when to use what files

### make_n4bc_array_script.py
This script is ran first to create the copy_segmentations.sh, the create_subtree.sh and the n4bc_array.sh files. These files are subsequently used for performing the n4 bias correction. In this file pay attention to the fact that there are many strings that need to be adapted to your specific purpose.

### copy_segmentations.sh
This script is used to copy the ground truths from a location prior to n4bc to a location after n4bc is conducted on the dataset.


### create_subtree.sh
This script is used to recreate the folder structure such that a similar structure is applied when using the new n4bc dataset.

### n4bc_array.sh
Here the commands for executing the bias correction are listed.