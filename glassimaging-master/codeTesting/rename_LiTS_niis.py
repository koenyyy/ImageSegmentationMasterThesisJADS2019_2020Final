import os
# mv file1.txt file2.txt

# script that corrects the wrong naming of the LiTS dataset as a result of improper ID storing

def get_gts(segmentations_loc):
    # Listing all the names of the segmentations
    gts = []  # create an empty list with segmentations

    for dirName, subdirList, fileList in os.walk(segmentations_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                if "segmentation" in filename.lower():
                    gts.append(os.path.join(dirName, filename))
    gts.sort()
    return gts

def get_segmentations(segmentations_loc):
    # Listing all the names of the segmentations
    segmentations = []  # create an empty list with segmentations

    for dirName, subdirList, fileList in os.walk(segmentations_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                if "segmented" in filename.lower():
                    segmentations.append(os.path.join(dirName, filename))
    segmentations.sort()
    return segmentations

def create_new_name_list(segmentations_loc, gts):
    old_name_list = segmentations_loc
    new_name_list = []
    for i in old_name_list:
        cur_file_int = int(i.split(os.sep)[-1].split('_')[0].lstrip('0'))-1
        gts_int = gts[cur_file_int].split(os.sep)[-1].split('-')[1].split('.nii')[0]
        new_file_name = i.replace(i.split(os.sep)[-1], gts_int + '_wasPrev_' + i.split(os.sep)[-1].split('_')[0] + '_segmented.nii.gz')
        new_name_list.append(new_file_name)
    return new_name_list

def rename_files(old_names, new_names):
    for i, j in zip(old_names, new_names):
        print('mv {0} {1}'.format(i, j))
        # os.system('mv {0} {1}'.format(i, j))

def run(gt_loc, segmentations_loc_list):
    for segmentations_loc in segmentations_loc_list:
        # get tge ids form the ground truths
        gts = get_gts(gt_loc)
        # get the ids from the segmented images that were named wrongly
        segmentations = get_segmentations(segmentations_loc)

        new_name_list = create_new_name_list(segmentations, gts)
        rename_files(segmentations, new_name_list)

if __name__ == '__main__':
    gt_loc = "D:\Thesis\Data\Lits17 unzipped"
    segmentations_loc_list = ["D:\Thesis\Data\lits segmentations for testing",
                              "D:\Thesis\Data\lits segmentations for testing"]

    run(gt_loc, segmentations_loc_list)


