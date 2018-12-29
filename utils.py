import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_moons_plot(epoch, best_model, dataset):
    # generate some examples
    best_model.eval()
    u = np.random.randn(500, 2).astype(np.float32)
    u_tens = torch.from_numpy(u).to(next(best_model.parameters()).device)
    x_synth = best_model.forward(u_tens, mode='inverse')[0].detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:,0], dataset.val.x[:,1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:,0], x_synth[:,1], '.')
    ax.set_title('Synth data')
    
    try:
        os.makedirs('plots')
    except OSError:
        pass
    
    plt.savefig('plots/plot_{:03d}.png'.format(epoch))
    plt.close()
