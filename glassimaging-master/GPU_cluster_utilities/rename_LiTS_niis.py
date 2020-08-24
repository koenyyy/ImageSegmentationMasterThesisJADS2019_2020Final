import os
# mv file1.txt file2.txt

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
        #os.system('mv {0} {1}'.format(i, j))

def run(gt_loc, segmentations_loc_list):
    for segmentations_loc in segmentations_loc_list:
        # get tge ids form the ground truths
        gts = get_gts(gt_loc)
        # get the ids from the segmented images that were named wrongly
        segmentations = get_segmentations(segmentations_loc)

        new_name_list = create_new_name_list(segmentations, gts)
        rename_files(segmentations, new_name_list)

if __name__ == '__main__':
    gt_loc = "/media/data/kderaad/LiTSRes1"
    segmentations_loc_list = ["/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200601160558_LiTS_zscore_noOtsu_noBC_Res1/eval_nifti_20200601160558_LiTS_zscore_noOtsu_noBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200601160638_LiTS_iscaling_noOtsu_noBC_Res1/eval_nifti_20200601160638_LiTS_iscaling_noOtsu_noBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200601160706_LiTS_iscaling_withOtsu_noBC_Res1/eval_nifti_20200601160706_LiTS_iscaling_withOtsu_noBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200601160750_LiTS_zscore_withOtsu_noBC_Res1/eval_nifti_20200601160750_LiTS_zscore_withOtsu_noBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200603102440_LiTS_zscore_noOtsu_withBC_Res1/eval_nifti_20200603102440_LiTS_zscore_noOtsu_withBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200603102513_LiTS_iscaling_noOtsu_withBC_Res1/eval_nifti_20200603102513_LiTS_iscaling_noOtsu_withBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200603102549_LiTS_iscaling_withOtsu_withBC_Res1/eval_nifti_20200603102549_LiTS_iscaling_withOtsu_withBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200603102629_LiTS_zscore_withOtsu_withBC_Res1/eval_nifti_20200603102629_LiTS_zscore_withOtsu_withBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181431_LiTS_zscore_noOtsu_noBC_Res2/eval_nifti_20200605181431_LiTS_zscore_noOtsu_noBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181527_LiTS_iscaling_noOtsu_noBC_Res2/eval_nifti_20200605181527_LiTS_iscaling_noOtsu_noBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181547_LiTS_iscaling_withOtsu_noBC_Res2/eval_nifti_20200605181547_LiTS_iscaling_withOtsu_noBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181615_LiTS_zscore_withOtsu_noBC_Res2/eval_nifti_20200605181615_LiTS_zscore_withOtsu_noBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181753_LiTS_zscore_noOtsu_noBC_Res4/eval_nifti_20200605181753_LiTS_zscore_noOtsu_noBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181842_LiTS_iscaling_noOtsu_noBC_Res4/eval_nifti_20200605181842_LiTS_iscaling_noOtsu_noBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181904_LiTS_iscaling_withOtsu_noBC_Res4/eval_nifti_20200605181904_LiTS_iscaling_withOtsu_noBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605181938_LiTS_zscore_withOtsu_noBC_Res4/eval_nifti_20200605181938_LiTS_zscore_withOtsu_noBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182105_LiTS_zscore_noOtsu_withBC_Res2/eval_nifti_20200605182105_LiTS_zscore_noOtsu_withBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182229_LiTS_iscaling_noOtsu_withBC_Res2/eval_nifti_20200605182229_LiTS_iscaling_noOtsu_withBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182320_LiTS_iscaling_withOtsu_withBC_Res2/eval_nifti_20200605182320_LiTS_iscaling_withOtsu_withBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182355_LiTS_zscore_withOtsu_withBC_Res2/eval_nifti_20200605182355_LiTS_zscore_withOtsu_withBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182556_LiTS_zscore_noOtsu_withBC_Res4/eval_nifti_20200605182556_LiTS_zscore_noOtsu_withBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182634_LiTS_iscaling_noOtsu_withBC_Res4/eval_nifti_20200605182634_LiTS_iscaling_noOtsu_withBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182655_LiTS_iscaling_withOtsu_withBC_Res4/eval_nifti_20200605182655_LiTS_iscaling_withOtsu_withBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200605182721_LiTS_zscore_withOtsu_withBC_Res4/eval_nifti_20200605182721_LiTS_zscore_withOtsu_withBC_Res4",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200609100121_LiTS_noNorm_noOtsu_noBC_Res1/eval_nifti_20200609100121_LiTS_noNorm_noOtsu_noBC_Res1",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200609100233_LiTS_noNorm_noOtsu_noBC_Res2/eval_nifti_20200609100233_LiTS_noNorm_noOtsu_noBC_Res2",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200609100831_LiTS_noNorm_noOtsu_noBC_Res4/eval_nifti_20200609100831_LiTS_noNorm_noOtsu_noBC_Res4"]

    run(gt_loc, segmentations_loc_list)


