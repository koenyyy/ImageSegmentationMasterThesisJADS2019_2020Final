import SimpleITK as sitk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from glassimaging.execution.jobs.jobtrain import JobTrain
import cv2

# Here we tested multiple preprocessing algorithms to find out where most time went. (using np.mean and np.std that is)

#
# print('start')
# t1 = datetime.now()
# img = sitk.ReadImage('C:\\\\Users\\\\s145576\\\\Documents\\\\GitHub\\\\automaticSegmentationThesis\\\\data\\\\intermediate_data\\\\volume-1.nii\\\\volume-1.nii')
# t2 = datetime.now()
#
# img_np = sitk.GetArrayFromImage(img)
# t3 = datetime.now()
#
# # use z-scoring
# print('Normalizing usign the z-score')
# values_nonzero = img_np[np.nonzero(img_np)]
# t4 = datetime.now()
# mean_nonzero = np.mean(values_nonzero)
# t5 = datetime.now()
# std_nonzero = np.std(values_nonzero)
# t6 = datetime.now()
#
# if std_nonzero == 0:
#     raise ValueError('Standard deviation of image is zero')
# img_n = (img_np - mean_nonzero) / std_nonzero
# t7 = datetime.now()
# print('end of file')
#
# print("Read image Took:", t2-t1)
# print("Get array from image Took:", t3-t2)
# print("Get non-zero Took:", t4-t3)
# print("Get mean Took:", t5-t4)
# print("Get std Took:", t6-t5)
# print("Z-scoring Took:", t7-t6)
#
# fig, axs = plt.subplots(3, 1)
#
# axs[0].imshow(img_n[400, :, :], cmap='Greys')
# axs[1].imshow(img_n[500, :, :], cmap='Greys')
# axs[2].imshow(img_n[600, :, :], cmap='Greys')
#
# plt.plot()
# plt.show()

# configfile, name, tmpdir, homedir = 'config/train_unet.json', 'eval', 'experiment_results', 'experiment_results'
# (torchloader, testloader) = JobTrain(configfile, name, tmpdir, homedir).getDataloader()
# for i_batch, sample_batched in enumerate(torchloader):
#     print(i_batch)

def test_non_zero_vs_other(img_path):
    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)


    t1 = datetime.now()
    values_nonzero = img_np[np.nonzero(img_np)]
    t2 = datetime.now()
    flat_img = img_np.flatten()
    # values_nonzero2 = flat_img[np.fliplr(cv2.findNonZero((flat_img > 0).astype(np.uint8)).squeeze())[:,0]]
    # values_nonzero2 = np.fliplr(cv2.findNonZero(img_np.flatten()))
    t3 = datetime.now()

    mean_nonzero = np.mean(values_nonzero)
    t4 = datetime.now()
    std_nonzero = np.std(values_nonzero)
    t5 = datetime.now()
    img_n = (img_np - mean_nonzero) / std_nonzero
    t6 = datetime.now()

    # mean_nonzero_cv, std_nonzero_cv = cv2.meanStdDev(values_nonzero2)
    t7 = datetime.now()
    # img_n2 = (img_np - mean_nonzero_cv[0][0]) / std_nonzero_cv[0][0]
    t8 = datetime.now()

    # print('means (np, cv):', np.mean(values_nonzero2), cv2.mean(values_nonzero2))
    # print('stds (np, cv):', np.std(values_nonzero2), cv2.meanStdDev(values_nonzero2))
    # print('meancv stdcv', mean_nonzero_cv, std_nonzero_cv)
    print('time for np\'s nonzero:', t2 - t1)
    print('time for alternative:', t3 - t2)

    print('time for np mean', t4 - t3)
    print('time for np std', t5 - t4)
    print('time for zscoring', t6 - t5)
    print('time for cv mean&std', t7 - t6)
    print('time for zscoring', t8 - t7)
    print('norm time:', t8-t1)


image_list = ['C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data\\volume-0.nii\\volume-0.nii',
              'C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data\\volume-1.nii\\volume-1.nii',
              'C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data\\volume-2.nii\\volume-2.nii',
              'C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data\\volume-3.nii\\volume-3.nii',
              'C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data\\volume-4.nii\\volume-4.nii']
for image in image_list:
    test_non_zero_vs_other(image)

test_non_zero_vs_other(image_list[0])
