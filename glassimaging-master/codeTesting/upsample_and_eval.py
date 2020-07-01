import SimpleITK as sitk
import torch
import pandas as pd
import os
from os import walk
from glassimaging.models.diceloss import DiceLoss
from glassimaging.evaluation.utils import getPerformanceMeasures
import numpy as np

# Code that is used to upsample smaller sized images as a result of using a resampling factor. After upsampling
# the upsampled images can be evaluated. 

def save_resampled_image(img, result_path, seg_id):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    result_path = os.path.join(result_path, '{}_segmented.nii.gz'.format(seg_id))
    sitk.WriteImage(img, result_path)

def upsample(segmentation, reference_img, seg_id):
    factor = [ref/seg for ref, seg in zip(reference_img.GetSize(), segmentation.GetSize())]
    segmentation.SetSpacing(factor)
    reference_image2 = sitk.Image(reference_img.GetSize(), reference_img.GetPixelIDValue())
    reference_image2.SetOrigin(reference_img.GetOrigin())
    reference_image2.SetDirection(reference_img.GetDirection())
    reference_image2.SetSpacing(reference_img.GetSpacing())

    resampled_seg = sitk.Resample(segmentation, reference_image2)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(dir_path, 'upsampled_segmentations', seg_id)
    save_resampled_image(resampled_seg, result_path, seg_id)

    return resampled_seg

def eval(segmentation, target, results, seg_id):
    criterion = DiceLoss()
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    target_array = sitk.GetArrayFromImage(target)

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

def run_upsample_and_eval(seg_loc, ground_truth_loc, seg_id, results):
    seg = sitk.ReadImage(seg_loc)
    ground_truth = sitk.ReadImage(ground_truth_loc)
    # print("1", seg.GetSize(), ground_truth.GetSize())
    # if "Res2" in seg_loc or "Res4" in seg_loc:
    #     resampling_factor = int(seg_loc.split("_Res")[1][0])
    #     print("resampling_factor is :", resampling_factor)
    #
    #     seg_np = sitk.GetArrayFromImage(seg)
    #     ground_truth_np = sitk.GetArrayFromImage(ground_truth)
    #
    #     seg_size = seg_np.shape
    #     gt_size = ground_truth_np.shape
    #
    #     if not seg_size == tuple(int(i / resampling_factor) for i in gt_size):
    #         # here we calculate the difference in size and divide that by 2 to get the number that needs to be substracted from both sides in a dimension
    #         size_diff = tuple((i - int(j / resampling_factor)) for i, j in zip(seg_size, gt_size))
    #         seg_np_normal_size = seg_np[size_diff[0]:, size_diff[1]:, size_diff[2]:]
    #         print('2', size_diff, seg_np_normal_size.shape)
    #
    #         # make a copy of needed image information before creating image from np array
    #         seg_origin = seg.GetOrigin()
    #         seg_direction = seg.GetDirection()
    #         seg_spacing = seg.GetSpacing()
    #         seg = sitk.GetImageFromArray(seg_np_normal_size)
    #         # assign earlier stored information back to new image
    #         seg.SetOrigin(seg_origin)
    #         seg.SetDirection(seg_direction)
    #         seg.SetSpacing(seg_spacing)

    upsampled_seg = upsample(seg, ground_truth, seg_id)
    print('image upsampled to size', upsampled_seg.GetSize(), 'vx spacing', upsampled_seg.GetSpacing())
    results = eval(upsampled_seg, ground_truth, results, seg_id)
    return results

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


def get_ground_truths(ground_truths_loc):
    # Listing all the names of the segmentations
    ground_truth = []  # create an empty list with segmentations

    for dirName, subdirList, fileList in os.walk(ground_truths_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                if "segmentation" in filename.lower():
                    ground_truth.append(os.path.join(dirName, filename))

    ground_truth.sort()
    return ground_truth


def get_overlapping_segs(segmentations, ground_truths):
    available_ground_truths = []

    # getting ids for all the segmentations made
    id_seg_list = []
    for seg in segmentations:
        id = seg.split(os.sep)[-1].split("_")[0]
        id_seg_list.append(id)

    # use the ids to find what ground truths are available and store them in a new list
    available_ground_truths = [gt for gt in ground_truths for id in id_seg_list if id in gt]

    # get ids for all the new_ground_truths
    id_gt_list = []
    for gt in available_ground_truths:
        id = gt.split(os.sep)[-2]
        id_gt_list.append(id)
    # use the ids present in the available_ground_truths list to find what segs can be used
    available_segmentations = [aseg for aseg in segmentations for id in id_gt_list if id in aseg]

    available_ground_truths.sort()
    id_gt_list.sort()
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
        results = run_upsample_and_eval(seg_loc, ground_truth_loc, id, results)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(os.path.join(dir_path, 'upsampled_segmentations')):
        os.makedirs(os.path.join(dir_path, 'upsampled_segmentations'))
    result_path = os.path.join(dir_path, 'upsampled_segmentations', 'upsampled_eval_results.csv')
    results.to_csv(result_path)
    return results

if __name__ == '__main__':
    segmentation_loc = "C:/Users/s145576/Documents/.Koen de Raad/year19.20/Thesis/Erasmus MC/Results/results for upsample and eval test/LiTS_Res4"
    ground_truths_loc = "C:/Users/s145576/Documents/.Koen de Raad/year19.20/Thesis/Erasmus MC/Results/results for upsample and eval test/LiTS_Res4"
    results = eval_all_segmentations(segmentation_loc, ground_truths_loc)
    print(results)
