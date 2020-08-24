import os
import numpy as np
import SimpleITK as sitk


def get_img_and_seg_lists(data_loc):
    # Listing all the names of the images and segmentations
    lstFilesGz = []  # create an empty list with images
    lstFilesGzSeg_femur = []  # create an empty list with segmentations of femur
    lstFilesGzSeg_fibula = []  # create an empty list with segmentations of femur
    lstFilesGzSeg_patella = []  # create an empty list with segmentations of femur
    lstFilesGzSeg_tibia = []  # create an empty list with segmentations of femur

    for dirName, subdirList, fileList in os.walk(data_loc):
        for filename in fileList:
            print(filename)
            if ".nii" in filename.lower():  # check whether the file's .nii
                # check whether a file has a segmentation from a specific person
                if "T1_T2_Morphology" in filename.lower():
                    lstFilesGz.append(os.path.join(dirName, filename))
                if "mask" in filename.lower() and "femur" in filename.lower():
                    lstFilesGzSeg_femur.append(os.path.join(dirName, filename))
                if "mask" in filename.lower() and "fibula" in filename.lower():
                    lstFilesGzSeg_fibula.append(os.path.join(dirName, filename))
                if "mask" in filename.lower() and "patella" in filename.lower():
                    lstFilesGzSeg_patella.append(os.path.join(dirName, filename))
                if "mask" in filename.lower() and "tibia" in filename.lower():
                    lstFilesGzSeg_tibia.append(os.path.join(dirName, filename))

    lstFilesGz.sort()
    lstFilesGzSeg_femur.sort()
    lstFilesGzSeg_fibula.sort()
    lstFilesGzSeg_patella.sort()
    lstFilesGzSeg_tibia.sort()

    return lstFilesGz, lstFilesGzSeg_femur, lstFilesGzSeg_fibula, lstFilesGzSeg_patella, lstFilesGzSeg_tibia


def merge_segs(seg_group):
    fem_seg = sitk.ReadImage(seg_group[0])
    fib_seg = sitk.ReadImage(seg_group[1])
    pat_seg = sitk.ReadImage(seg_group[2])
    tib_seg = sitk.ReadImage(seg_group[3])

    fem_seg_np = sitk.GetArrayFromImage(fem_seg)
    fib_seg_np = sitk.GetArrayFromImage(fib_seg)
    pat_seg_np = sitk.GetArrayFromImage(pat_seg)
    tib_seg_np = sitk.GetArrayFromImage(tib_seg)

    # give the segmentations different values before combining them
    # fem_seg_np[fem_seg_np == 1] = 1 # of course this one is redundant
    fib_seg_np[fib_seg_np == 1] = 2
    pat_seg_np[pat_seg_np == 1] = 3
    tib_seg_np[tib_seg_np == 1] = 4

    # here we take 3 steps overlaying the segmentations ontop of each other
    new_arr = np.where(fib_seg_np == 0, fem_seg_np, fib_seg_np)
    new_arr = np.where(pat_seg_np == 0, new_arr, pat_seg_np)
    new_arr = np.where(tib_seg_np == 0, new_arr, tib_seg_np)

    combined_segmentations = sitk.GetImageFromArray(new_arr)
    combined_segmentations.SetOrigin(fem_seg.GetOrigin())
    combined_segmentations.SetDirection(fem_seg.GetDirection())
    combined_segmentations.SetSpacing(fem_seg.GetSpacing())

    return combined_segmentations


def save_combined_seg(combined_seg, data_loc, seg_group):
    patient_id = seg_group[0].split(os.sep)[-1].split("_")[3]
    save_file = os.path.join(data_loc, 'masks', "combined_bone_mask_" + patient_id + '.nii.gz')
    sitk.WriteImage(combined_seg, save_file)

if __name__ == '__main__':
    data_loc = "/media/data/kderaad/Ergo_Res1"
    img_list, fem_seg_list, fib_seg_list, pat_seg_list, tib_seg_list = get_img_and_seg_lists(data_loc)

    for seg_group in zip(fem_seg_list, fib_seg_list, pat_seg_list, tib_seg_list):
        combined_seg = merge_segs(seg_group)
        save_combined_seg(combined_seg, data_loc, seg_group)
        print(seg_group)
