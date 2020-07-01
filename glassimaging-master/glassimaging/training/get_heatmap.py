import torch
import matplotlib.pyplot as plt

def get_heatmap(net_output):
    # np_output = net_output.detach().cpu().numpy()
    np_output = net_output
    print("######### THE SIZE IS: ", np_output.shape)

    fig, axs = plt.subplots(2, 6)
    axs[0, 0].imshow(np_output[0, :, :, 100])
    axs[1, 0].imshow(np_output[0, :, :, 118])

    axs[0, 1].imshow(np_output[0, :, :, 103])
    axs[1, 1].imshow(np_output[0, :, :, 121])

    axs[0, 2].imshow(np_output[0, :, :, 106])
    axs[1, 2].imshow(np_output[0, :, :, 124])

    axs[0, 3].imshow(np_output[0, :, :, 109])
    axs[1, 3].imshow(np_output[0, :, :, 127])

    axs[0, 4].imshow(np_output[0, :, :, 112])
    axs[1, 4].imshow(np_output[0, :, :, 130])

    axs[0, 5].imshow(np_output[0, :, :, 115])
    axs[1, 5].imshow(np_output[0, :, :, 133])

    plt.plot()
    plt.show()

