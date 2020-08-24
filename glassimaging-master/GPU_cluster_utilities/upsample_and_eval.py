import SimpleITK as sitk
import torch
import pandas as pd
import os
from os import walk
from glassimaging.models.diceloss import DiceLoss
from glassimaging.evaluation.utils import getPerformanceMeasures
import numpy as np

# for all experiments that made use of a resampling factor of 2 or 4 the evals are too small and need upsampling to be properly evaluated. This script does just that. Here we upsample the eval results and re-evaluate them again against the ground truth to find out if using a resampling factor was actually beneficial. 

def save_resampled_image(img, result_path, seg_id):
    result_path = os.path.join(result_path, '{}_upsampled_segmented.nii.gz'.format(seg_id))
    sitk.WriteImage(img, result_path)

def upsample(segmentation, reference_img, seg_id, segmentations_loc):
    factor = [ref/seg for ref, seg in zip(reference_img.GetSize(), segmentation.GetSize())]
    segmentation.SetSpacing(factor)
    reference_image2 = sitk.Image(reference_img.GetSize(), reference_img.GetPixelIDValue())
    reference_image2.SetOrigin(reference_img.GetOrigin())
    reference_image2.SetDirection(reference_img.GetDirection())
    reference_image2.SetSpacing(reference_img.GetSpacing())

    resampled_seg = sitk.Resample(segmentation, reference_image2)
	# save an image depending on the ID
    if seg_id == "0272":
        print('saved upsampled version of 0272')
        save_resampled_image(resampled_seg, segmentations_loc, seg_id)

    return resampled_seg

def eval(segmentation, target, results, seg_id):
    criterion = DiceLoss()
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    target_array = sitk.GetArrayFromImage(target)

    print(segmentation_array.shape, target_array.shape)
    if results.empty:
        sample_num = 0
    else:
        sample_num = results['sample'].iloc[-1]

    for c in range(0, 5):
        truth = target_array == c
        positive = segmentation_array == c
        (dice, TT, FP, FN, TN) = getPerformanceMeasures(positive, truth)
        results = results.append(
            {'sample': sample_num, 'class': c, 'subject': seg_id, 'TP': TT, 'FP': FP, 'FN': FN, 'TN': TN,
             'dice': dice}, ignore_index=True)

    return results

def run_upsample_and_eval(seg_loc, ground_truth_loc, seg_id, results, segmentations_loc):
    seg = sitk.ReadImage(seg_loc)
    ground_truth = sitk.ReadImage(ground_truth_loc)
    #print(seg_loc, ground_truth_loc)
    upsampled_seg = upsample(seg, ground_truth, seg_id, segmentations_loc)
    #print('image upsampled to size', upsampled_seg.GetSize(), 'vx spacing', upsampled_seg.GetSpacing())
    results = eval(upsampled_seg, ground_truth, results, seg_id)
    return results

def get_segmentations(segmentations_loc):
    # Listing all the names of the segmentations
    segmentations = []  # create an empty list with segmentations

	# make sure that segmented is the right indicator to find your segmentations. this may not always be the case
    for dirName, subdirList, fileList in os.walk(segmentations_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                if "segmented" in filename.lower() and not "0272_upsampled_segmented" in filename.lower():
                    segmentations.append(os.path.join(dirName, filename))
    segmentations.sort()
    return segmentations


def get_ground_truths(ground_truths_loc):
    # Listing all the names of the segmentations
    ground_truth = []  # create an empty list with segmentations

	# make sure that mask is the right indicator to find your segmentations. this may not always be the case
    for dirName, subdirList, fileList in os.walk(ground_truths_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                if "mask" in filename.lower() and ("23" in filename.lower() or "21" in filename.lower()):
                    ground_truth.append(os.path.join(dirName, filename))

    ground_truth.sort()
    return ground_truth


def get_overlapping_segs(segmentations, ground_truths):
    available_ground_truths = []

    # getting ids for all the segmentations made
    id_seg_list = []
    for seg in segmentations:
        id = seg.split(os.sep)[-1].split("-")[1].split("_")[0]
        id_seg_list.append(id)

    # use the ids to find what ground truths are available and store them in a new list
    available_ground_truths = [gt for gt in ground_truths for id in id_seg_list if id in gt.split(os.sep)[-2].split("-")[1]]

    # get ids for all the new_ground_truths
    id_gt_list = []
    for gt in available_ground_truths:
        id = gt.split(os.sep)[-2].split("-")[1]
        id_gt_list.append(id)
    # use the ids present in the available_ground_truths list to find what segs can be used
    available_segmentations = [aseg for aseg in segmentations for id in id_gt_list if id in aseg.split(os.sep)[-1].split("-")[1].split("_")[0]]

    available_segmentations.sort()
    available_ground_truths.sort()
    id_gt_list.sort()
    print('#1', len(segmentations), segmentations, len(id_gt_list), id_gt_list)
    print('seg', id_seg_list, '\n', 'gt', id_gt_list)
    print('available_segmentations', len(available_segmentations), available_segmentations)
    print('available_ground_truths', len(available_ground_truths), available_ground_truths)
    print('these files will be upsampled and evaluated', id_gt_list)

    return zip(available_segmentations, available_ground_truths), id_gt_list

def eval_all_segmentations(segmentations_loc, ground_truths_loc):
    results = pd.DataFrame(columns=['sample', 'class', 'subject', 'TP', 'FP', 'FN', 'TN', 'dice'])

    # get sorted list of segmentations and ground_truths
    segmentations = get_segmentations(segmentations_loc)
    ground_truths = get_ground_truths(ground_truths_loc)

    # get the ground truths that belong to the segmentations
    overlapping_segs, id_list = get_overlapping_segs(segmentations, ground_truths)

    # for all segmentations get the corresponding ground truth and run the upsample and eval procedure
    for (seg_loc, ground_truth_loc), id in zip(overlapping_segs, id_list):
        results = run_upsample_and_eval(seg_loc, ground_truth_loc, id, results, segmentations_loc)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(segmentations_loc, 'upsampled_eval_results.csv')
    results.to_csv(result_path)
    return results

if __name__ == '__main__':
# before running make sure that all strings in the document are tailored for your dataset. for example, some datasets name their ground truths without the word "mask" but with "GT"
    list_of_folders_to_upsample_and_eval = ["/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082733_BTD_zscore_withOtsu_withBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082753_BTD_zscore_withOtsu_noBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082813_BTD_zscore_withOtsu_withBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082834_BTD_zscore_withOtsu_noBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082934_BTD_zscore_noOtsu_withBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619082954_BTD_zscore_noOtsu_noBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083014_BTD_zscore_noOtsu_withBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083034_BTD_zscore_noOtsu_noBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083135_BTD_iscaling_withOtsu_withBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083155_BTD_iscaling_withOtsu_noBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083215_BTD_iscaling_withOtsu_withBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083235_BTD_iscaling_withOtsu_noBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083335_BTD_iscaling_noOtsu_withBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083355_BTD_iscaling_noOtsu_noBC_Res2_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083415_BTD_iscaling_noOtsu_withBC_Res4_Seed364/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200619083436_BTD_iscaling_noOtsu_noBC_Res4_Seed364/eval_nifti/result"]
    for folder in list_of_folders_to_upsample_and_eval:
        segmentations_loc = folder
        ground_truths_loc = "/media/data/kderaad/BTD_N4BC"
        results = eval_all_segmentations(segmentations_loc, ground_truths_loc)