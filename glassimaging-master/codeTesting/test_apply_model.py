import matplotlib.pyplot as plt
import torch
import os
from torch import optim
from glassimaging.models.ResUNet3D import ResUNet
import sys
from mpl_toolkits.axes_grid1 import ImageGrid
import SimpleITK as sitk
import numpy as np

# Code that allows for a model to be used to pass an image through. Only works using the GPU cluster as local
# GPU is too low on memory

def saveImageBatch(sample_batched, frameNr, batchNr):
    """Save image with segmentation for a batch of samples."""
    images_batch, segmentations_batch, ground_truth_batch = \
        sample_batched['image'], sample_batched['segmentation'], sample_batched['ground_truth']

    batch_size = len(images_batch)
    num_classes = len(segmentations_batch[0])

    fig = plt.figure(figsize=(18., 9.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(batch_size, 2 * num_classes),  # creates grid that has dimension of (plain img + classes + ground truths) x batch size of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    # getting to the right folder and trained network
    experiment_results_loc = os.path.join(sys.path[0], 'experiment_results')
    experiment_results_loc = next(os.walk(experiment_results_loc))

    for i in range(batch_size):
        # create list to store images of single batch
        im_list = []
        # append normal image that needs to be segmented as well as ground truth
        im_list.append(images_batch[i][0][frameNr])

        # for each segmented class and ground truth class add corresponding image
        for index, j in enumerate(range(num_classes)):
            if index == 0:
                im_list.append(segmentations_batch[i][j][frameNr])
            else:
                # TODO zorgen dat ook voor segmentaties waar de ground truth slechts 1 slice is het kan worden gedisplayed
                im_list.append(ground_truth_batch[i][j-1][frameNr])
                im_list.append(segmentations_batch[i][j][frameNr])

        # plot the image of a single batch
        for k, (ax, im) in enumerate(zip(grid, im_list)):
            # Iterating over the grid returns the Axes.
            if k == 0:
                ax.imshow(im, cmap='gray')
            # ik k is even, we're dealing with a ground truth
            elif k % 2 == 0:
                ax.imshow(im_list[0].squeeze(), cmap='gray')
                ax.imshow(im, cmap='Reds', alpha=0.4, interpolation='none')
            else:
                ax.imshow(im_list[0].squeeze(), cmap='gray')
                masked_seg = np.ma.masked_where(im.squeeze() < 0.05, im.squeeze())
                ax.imshow(masked_seg, cmap='Reds', alpha=0.4, interpolation='none')

        latest_trained_net_loc = os.path.join(os.path.join(experiment_results_loc[0], experiment_results_loc[1][-1]),
                                              'images', 'segmented_batch_{nr}.png'.format(nr=batchNr))
        plt.savefig(latest_trained_net_loc)

path = 'C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\24ExperimentsBTD\\Results BTD 1 - 4\\BTD_zscore_withOtsu_noBC_Res1\\model.pt'
checkpoint = torch.load(path)

model = ResUNet(k=32, outputsize=2, inputsize=1).float()

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.eval()

with torch.no_grad():
    img_1 = sitk.ReadImage('C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Data\\BTD\\BTD-0002\\2310.nii.gz')


    img_1 = sitk.GetArrayFromImage(img_1)


    img_1 = torch.from_numpy(img_1)
    img_1 = img_1.unsqueeze(0).unsqueeze(0)


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)  # send network to GPU device if available
    # img_1 = img_1.float().to(device)

    print('running on: cpu')
    print(type(img_1))
    print(img_1.device)

    output = model(img_1)

    output = output.numpy()

    print(output.shape)

    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(output[0, 0, 30, :, :])
    axs[1].imshow(output[0, 0, 80, :, :])
    axs[2].imshow(output[0, 0, 130, :, :])

    plt.plot()
    plt.show()





    # # observe batch.
    # # displaying 20th slice
    # saveImageBatch(
    #     {'image': sample_batched['image'], 'segmentation': output, 'ground_truth': sample_batched['segmentation']},
    #         20, batchNr=i_batch)

