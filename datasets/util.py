import os
import pickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng


def isposint(n):
    """
    Determines whether number n is a positive integer.
    :param n: number
    :return: bool
    """
    return isinstance(n, int) and n > 0


def logistic(x):
    """
    Elementwise logistic sigmoid.
    :param x: numpy array
    :return: numpy array
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))


def disp_imdata(xs, imsize, layout=(1, 1)):
    """
    Displays an array of images, a page at a time. The user can navigate pages with
    left and right arrows, start over by pressing space, or close the figure by esc.
    :param xs: an numpy array with images as rows
    :param imsize: size of the images
    :param layout: layout of images in a page
    :return: none
    """

    num_plots = np.prod(layout)
    num_xs = xs.shape[0]
    idx = [0]

    # create a figure with suplots
    fig, axs = plt.subplots(layout[0], layout[1])

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    def plot_page():
        """Plots the next page."""

        ii = np.arange(idx[0], idx[0] + num_plots) % num_xs

        for ax, i in zip(axs, ii):
            ax.imshow(xs[i].reshape(imsize), cmap='gray', interpolation='none')
            ax.set_title(str(i))

        fig.canvas.draw()

    def on_key_event(event):
        """Event handler after key press."""

        key = event.key

        if key == 'right':
            # show next page
            idx[0] = (idx[0] + num_plots) % num_xs
            plot_page()

        elif key == 'left':
            # show previous page
            idx[0] = (idx[0] - num_plots) % num_xs
            plot_page()

        elif key == ' ':
            # show first page
            idx[0] = 0
            plot_page()

        elif key == 'escape':
            # close figure
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_event)
    plot_page()


def isdistribution(p):
    """
    :param p: a vector representing a discrete probability distribution
    :return: True if p is a valid probability distribution
    """
    return np.all(p >= 0.0) and np.isclose(np.sum(p), 1.0)


def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = rng.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)


def ess_importance(ws):
    """
    Calculates the effective sample size of a set of weighted independent samples (e.g. as given by importance
    sampling or sequential monte carlo). Takes as input the normalized sample weights.
    """

    ess = 1.0 / np.sum(ws**2)
    return ess


def ess_mcmc(xs):
    """
    Calculates the effective sample size of a correlated sequence of samples, e.g. as given by markov chain monte
    carlo.
    """

    n_samples, n_dim = xs.shape

    mean = np.mean(xs, axis=0)
    xms = xs - mean

    acors = np.zeros_like(xms)
    for i in range(n_dim):
        for lag in range(n_samples):
            acor = np.sum(xms[:n_samples - lag, i] * xms[lag:, i]) / (
                n_samples - lag)
            if acor <= 0.0: break
            acors[lag, i] = acor

    act = 1.0 + 2.0 * np.sum(acors[1:], axis=0) / acors[0]
    ess = n_samples / act

    return np.min(ess)


def probs2contours(probs, levels):
    """
    Takes an array of probabilities and produces an array of contours at specified percentile levels
    :param probs: probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    :param levels: percentile levels. have to be in [0.0, 1.0]
    :return: array of same shape as probs with percentile labels
    """

    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf_marginals(pdf, lims, gt=None, levels=(0.68, 0.95)):
    """
    Plots marginals of a pdf, for each variable and pair of variables.
    """

    if pdf.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(lims[0], lims[1], 200)

        pp = pdf.eval(xx[:, np.newaxis], log=False)
        ax.plot(xx, pp)
        ax.set_xlim(lims)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        fig, ax = plt.subplots(pdf.ndim, pdf.ndim)

        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf.ndim, 1]) if lims.ndim == 1 else lims

        for i in range(pdf.ndim):
            for j in range(pdf.ndim):

                if i == j:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 500)
                    pp = pdf.eval(xx, ii=[i], log=False)
                    ax[i, j].plot(xx, pp)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if gt is not None:
                        ax[i, j].vlines(
                            gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 200)
                    yy = np.linspace(lims[j, 0], lims[j, 1], 200)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]),
                         Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[i, j], log=False)
                    pp = pp.reshape(list(X.shape))
                    ax[i, j].contour(X, Y, probs2contours(pp, levels), levels)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax


def plot_hist_marginals(data, lims=None, gt=None):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, normed=True)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if lims is not None: ax.set_xlim(lims)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        n_dim = data.shape[1]
        fig, ax = plt.subplots(n_dim, n_dim)
        ax = np.array([[ax]]) if n_dim == 1 else ax

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):

                if i == j:
                    ax[i, j].hist(data[:, i], n_bins, normed=True)
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if lims is not None: ax[i, j].set_xlim(lims[i])
                    if gt is not None:
                        ax[i, j].vlines(
                            gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                        ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax


def save(data, file):
    """
    Saves data to a file.
    """

    f = open(file, 'w')
    pickle.dump(data, f)
    f.close()


def load(file):
    """
    Loads data from file.
    """

    f = open(file, 'r')
    data = pickle.load(f)
    f.close()
    return data


def calc_whitening_transform(xs):
    """
    Calculates the parameters that whiten a dataset.
    """

    assert xs.ndim == 2, 'Data must be a matrix'
    N = xs.shape[0]

    means = np.mean(xs, axis=0)
    ys = xs - means

    cov = np.dot(ys.T, ys) / N
    vars, U = np.linalg.eig(cov)
    istds = np.sqrt(1.0 / vars)

    return means, U, istds


def whiten(xs, params):
    """
    Whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys -= means
    ys = np.dot(ys, U)
    ys *= istds

    return ys


def copy_model_parms(source_model, target_model):
    """
    Copies the parameters of source_model to target_model.
    """

    for sp, tp in zip(source_model.parms, target_model.parms):
        tp.set_value(sp.get_value())


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)
