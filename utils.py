import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def save_moons_plot(epoch, best_model, dataset):
    # generate some examples
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(500).detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:, 0], dataset.val.x[:, 1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Synth data')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    plt.savefig('plots/plot_{:03d}.png'.format(epoch))
    plt.close()


fixed_noise = torch.Tensor(64, 28 * 28).normal_()

def save_images(epoch, best_model):
    best_model.eval()
    with torch.no_grad():
        imgs = best_model.sample(64, noise=fixed_noise).detach().cpu()
        imgs = torch.sigmoid(imgs.view(64, 1, 28, 28))
    
    try:
        os.makedirs('images')
    except OSError:
        pass

    torchvision.utils.save_image(imgs, 'images/img_{:03d}.png'.format(epoch), nrow=8)

