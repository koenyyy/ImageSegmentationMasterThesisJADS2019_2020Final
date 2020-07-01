import matplotlib.pyplot as plt
import torch
import os
from torch import optim
from glassimaging.models.ResUNet3D import ResUNet
import SimpleITK as sitk
import numpy as np

# Code that allows for a model to be used to pass an image through. Only works using the GPU cluster as local
# GPU is too low on memory

def resample_img2(image, reference_img, resampling_factor, use_seg=False):

    img_spacing = reference_img.GetSpacing()
    img_direction = reference_img.GetDirection()
    img_origin = reference_img.GetOrigin()
    img_size = reference_img.GetSize()
    img_pixelIDValue = reference_img.GetPixelIDValue()

    new_img_size = tuple(int(i / resampling_factor) for i in img_size)
    new_img_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_img_size, img_size, img_spacing)]

    resample_to_this_image = sitk.Image(*new_img_size, img_pixelIDValue)

    resample_to_this_image.SetSpacing(new_img_spacing)

    image_to_resample = sitk.GetImageFromArray(image)

    if not use_seg:
        resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkBSplineResamplerOrder3)
    else:
        resampled_img = sitk.Resample(image_to_resample, resample_to_this_image, sitk.Transform(), sitk.sitkNearestNeighbor)

    resampled_img_np = sitk.GetArrayFromImage(resampled_img)

    return resampled_img_np


path = 'C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\24ExperimentsBTD\\Results 1BTD 1 - 8 Res 1\\BTD_zscore_withOtsu_noBC_Res1\\model.pt'
checkpoint = torch.load(path)

model = ResUNet(k=32, outputsize=2, inputsize=1).float()

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.eval()

with torch.no_grad():
    img_1 = sitk.ReadImage('C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Data\\BTD\\BTD-0002\\2310.nii.gz')

    img_1 = sitk.GetArrayFromImage(img_1)

    # DO NOT FORGET TO SET THE RESAMPLING FACTOR!
    img_np = resample_img2(img_1, sitk.ReadImage('C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Data\\BTD\\BTD-0002\\2310.nii.gz'), 2, use_seg=False)

    img_1 = torch.from_numpy(img_1)
    img_1 = img_1.unsqueeze(0).unsqueeze(0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # send network to GPU device if available
    img_1 = img_1.float().to(device)

    output = model(img_1)

    output = output.numpy()

    print(output.shape)

    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(output[0, 0, 30, :, :])
    axs[1].imshow(output[0, 0, 80, :, :])
    axs[2].imshow(output[0, 0, 130, :, :])

    plt.savefig('/media/data/kderaad/inspection_files/soft_dice_img_BTD-0002', format='png')
    plt.close()
