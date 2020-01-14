import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

image_counter = 0


# functions to show an image
def imshow(img):
    global image_counter
    image_counter += 1

    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    filename = "imgs/savefig_{}.png".format(image_counter)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_latent_space(dataiter, images, labels, args):
    # Plot mesh grid from latent space
    numImgs = 30
    lo, hi = -3., 3.

    # Define mesh grid ticks
    z1 = torch.linspace(lo, hi, numImgs)
    z2 = torch.linspace(lo, hi, numImgs)

    # Create mesh as pair of elements
    z = []
    for idx in range(numImgs):
        for jdx in range(numImgs):
            z.append([z1[idx], z2[jdx]])
    z = torch.tensor(z).to(args.device)

    # Decode elements from latent space
    decoded_z = args.net.decode(z).cpu()

    # print images
    grid = torchvision.utils.make_grid(decoded_z, nrow=numImgs)
    imshow(grid)
    args.writer.add_image('latent-space-grid-decoded', grid)

    # Plot encoded test set into latent space
    numBatches = 500
    for idx in range(numBatches):
        tImages, tLabels = dataiter.next()
        images = torch.cat((images.cpu(), tImages.cpu()), 0)
        labels = torch.cat((labels.cpu(), tLabels.cpu()), 0)

    # encode into latent space
    images = images.cpu()
    encoded_images_loc, _ = args.net.cpu().encode(images)
    encoded_images_loc = encoded_images_loc.cpu().detach().numpy()

    # Scatter plot of latent space
    x = encoded_images_loc[:, 0]
    y = encoded_images_loc[:, 1]

    # Send to tensorboard
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    sct = ax.scatter(x, y, c=labels, cmap='jet')
    fig.colorbar(sct)
    args.writer.add_figure('scatter-plot-of-encoded-test-sample', fig)

    # Plot with matplotlib
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y, c=labels, cmap='jet')
    plt.colorbar()
    filename = "imgs/test_into_latent_space_{}.png".format(numBatches)
    plt.savefig(filename)
    plt.show()
