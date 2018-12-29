import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import datasets
import flows as fnn

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--dataset',
    default='POWER',
    help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
parser.add_argument('--flow', default='maf', help='flow to use: maf | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

assert args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS']
dataset = getattr(datasets, args.dataset)()

train_tensor = torch.from_numpy(dataset.trn.x)
train_dataset = torch.utils.data.TensorDataset(train_tensor)

valid_tensor = torch.from_numpy(dataset.val.x)
valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

test_tensor = torch.from_numpy(dataset.tst.x)
test_dataset = torch.utils.data.TensorDataset(test_tensor)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims
num_hidden = {
    'POWER': 100,
    'GAS': 100,
    'HEPMASS': 512,
    'MINIBOONE': 512,
    'BSDS300': 512,
    'MOONS': 64
}[args.dataset]

act = 'tanh' if args.dataset is 'GAS' else 'relu'

modules = []

assert args.flow in ['maf', 'glow']
for _ in range(args.num_blocks):
    if args.flow == 'glow':
        print("Warning: Results for GLOW are not as good as for MAF yet.")
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs),
            fnn.CouplingLayer(num_inputs, num_hidden, s_act='tanh', t_act='relu')
        ]
    elif args.flow == 'maf':
        modules += [
            fnn.MADE(num_inputs, num_hidden, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        module.bias.data.fill_(0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
        -1, keepdim=True)
    loss = -(log_probs + log_jacob).sum()
    if size_average:
        loss /= u.size(0)
    return loss


def train(epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        u, log_jacob = model(data)
        loss = flow_loss(u, log_jacob)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation'):
    model.eval()
    val_loss = 0

    for data in loader:
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            u, log_jacob = model(data)
            val_loss += flow_loss(
                u, log_jacob, size_average=False).item()  # sum up batch loss

    val_loss /= len(loader.dataset)
    print('\n{} set: Average loss: {:.4f}\n'.format(prefix, val_loss))

    return val_loss


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print('Best validation at epoch {}: Average loss: {:.4f}\n'.format(
        best_validation_epoch, best_validation_loss))

validate(best_validation_epoch, best_model, test_loader, prefix='Test')

if args.dataset == 'MOONS':
    # generate some examples
    best_model.eval()
    u = np.random.randn(500, 2).astype(np.float32)
    u_tens = torch.from_numpy(u).to(device)
    x_synth = best_model.forward(u_tens, mode='inverse')[0].detach().cpu().numpy()

    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:,0], dataset.val.x[:,1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:,0], x_synth[:,1], '.')
    ax.set_title('Synth data')

    plt.show()
