import argparse
import copy
import math
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import datasets
import flows as fnn
import utils


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
parser.add_argument('--flow', default='maf', help='flow to use: maf | realnvp | glow')
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

assert args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST']
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
    'MOONS': 64,
    'MNIST': 1024
}[args.dataset]

act = 'tanh' if args.dataset is 'GAS' else 'relu'

modules = []

assert args.flow in ['maf', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(num_inputs, num_hidden, mask, s_act='tanh', t_act='relu')
        ]
    mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()
    
    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(num_inputs, num_hidden, mask, s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
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

def train(epoch):
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        pbar.update(data.size(0))
        pbar.set_description(
                'Train, Log likelihood in nats: {:.6f}'.format(-train_loss / (batch_idx + 1)))
    pbar.close()

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

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description(
                'Val, Log likelihood in nats: {:.6f}'.format(-val_loss / pbar.n))

    pbar.close()
    return val_loss / len(loader.dataset)


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print('Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.format(
        best_validation_epoch, -best_validation_loss))

    if args.dataset == 'MOONS' and epoch % 10 == 0:
        utils.save_moons_plot(epoch, best_model, dataset)


validate(best_validation_epoch, best_model, test_loader, prefix='Test')
