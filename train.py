import torch
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.network import CSNet
from torch import nn
import time
import os

import argparse
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder
import torchvision.transforms as transforms
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--pre_epochs', default=200, type=int, help='pre train epoch number')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')

parser.add_argument('--batchSize', default=64, type=int, help='train batch size')
parser.add_argument('--sub_rate', default=0.1, type=float, help='sampling sub rate')

parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
BLOCK_SIZE = opt.block_size
NUM_EPOCHS = opt.num_epochs
PRE_EPOCHS = opt.pre_epochs
LOAD_EPOCH = 0


train_set = TrainDatasetFromFolder('/media/gdh-95/data/Train', crop_size=CROP_SIZE, blocksize=BLOCK_SIZE)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)

net = CSNet(BLOCK_SIZE, opt.sub_rate)

mse_loss = nn.MSELoss()

if opt.generatorWeights != '':
    net.load_state_dict(torch.load(opt.generatorWeights))
    LOAD_EPOCH = opt.loadEpoch

if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, }

    net.train()
    scheduler.step()

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = net(z)
        optimizer.zero_grad()
        g_loss = mse_loss(fake_img, real_img)

        g_loss.backward()
        optimizer.step()

        running_results['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(desc='[%d] Loss_G: %.4f lr: %.7f' % (
            epoch, running_results['g_loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))

    # for saving model
    save_dir = 'epochs' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(BLOCK_SIZE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % 5 == 0:
        torch.save(net.state_dict(), save_dir + '/net_epoch_%d_%6f.pth' % (epoch, running_results['g_loss']/running_results['batch_sizes']))

