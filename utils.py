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


batch_size = 100
fixed_noise = torch.Tensor(batch_size, 28 * 28).normal_()
y = torch.arange(batch_size).unsqueeze(-1) % 10
y_onehot = torch.FloatTensor(batch_size, 10)
y_onehot.zero_()
y_onehot.scatter_(1, y, 1)


def save_images(epoch, best_model, cond):
    best_model.eval()
    with torch.no_grad():
        if cond:
            imgs = best_model.sample(batch_size, noise=fixed_noise, cond_inputs=y_onehot).detach().cpu()
        else:
            imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()

        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))
    
    try:
        os.makedirs('images')
    except OSError:
        pass

    torchvision.utils.save_image(imgs, 'images/img_{:03d}.png'.format(epoch), nrow=10)

