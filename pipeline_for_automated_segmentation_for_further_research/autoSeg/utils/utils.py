import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

# def showImage(sample, frameNr):
#     plt.imshow(sample['image'][frameNr].squeeze(), cmap='gray')
#     print(sample['segmentation'].size())
#     masked_data = np.ma.masked_where(sample['segmentation'][frameNr].squeeze() < 0.05, sample['segmentation'][frameNr].squeeze())
#
#     plt.imshow(masked_data, cmap='Reds', alpha=0.4, interpolation='none')
#
#
# # Helper function to show a batch
# def showImageBatch(sample_batched, frameNr):
#     """Show image with segmentation for a batch of samples."""
#     images_batch, segmentations_batch = \
#         sample_batched['image'], sample_batched['segmentation']
#     print('len segmentations_batch', len(segmentations_batch), 'len images_batch:', len(images_batch))
#     print('size segmentations_batch', segmentations_batch.size(), 'size images_batch:', images_batch.size())
#     batch_size = len(images_batch)
#     num_classes = len(segmentations_batch[0])
#     im_size = images_batch.size(2)
#     grid_border_size = 2
#     print('num classes', num_classes)
#
#     fig = plt.figure(figsize=(15, 15))
#     for i in range(batch_size):
#         for j in range(num_classes):
#             ax = plt.subplot(num_classes, batch_size, (i*j) + (i + 1))
#             plt.tight_layout()
#             ax.axis('off')
#             showImage({'image': images_batch[i][0], 'segmentation': segmentations_batch[i][j]}, frameNr)
#     plt.show()

def showImageBatch(sample_batched, frameNr):
    """Show image with segmentation for a batch of samples."""
    images_batch, segmentations_batch, ground_truth_batch = \
        sample_batched['image'], sample_batched['segmentation'], sample_batched['ground_truth']

    batch_size = len(images_batch)
    num_classes = len(segmentations_batch[0])

    fig = plt.figure(figsize=(18., 9.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(batch_size, 2 * num_classes),  # creates grid that has dimension of (plain img + classes + ground truths) x batch size of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

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
        plt.show()

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


def dice_loss(pred, truth):
    pred = torch.sigmoid(pred)
    smooth = 1.
    print(pred.size(), truth.size())
    iflat = pred.view(-1)
    tflat = truth.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

