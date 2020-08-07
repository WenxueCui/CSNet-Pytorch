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

from data_utils import TestDatasetFromFolder, psnr
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.transforms import ToPILImage


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--save_img', default=1, type=int, help='')

parser.add_argument('--sub_rate', default=0.1, type=float, help='sampling sub rate')

parser.add_argument('--NetWeights', type=str, default='epochs_subrate_0.1_blocksize_32/net_epoch_200_0.001724.pth', help="path of CSNet weights for testing")

opt = parser.parse_args()

BLOCK_SIZE = opt.block_size

val_set = TestDatasetFromFolder('/media/gdh-95/data/Set14', blocksize=BLOCK_SIZE)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

net = CSNet(BLOCK_SIZE, opt.sub_rate)
mse_loss = nn.MSELoss()

if opt.NetWeights != '':
    net.load_state_dict(torch.load(opt.NetWeights))

if torch.cuda.is_available():
    net.cuda()
    mse_loss.cuda()

for epoch in range(1, 1+1):
    train_bar = tqdm(val_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0, }

    save_dir = 'results' + '_subrate_' + str(opt.sub_rate) + '_blocksize_' + str(
        BLOCK_SIZE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net.eval()
    psnrs = 0.0
    img_id = 0

    for data, target in train_bar:
        batch_size = data.size(0)
        if batch_size <= 0:
            continue

        running_results['batch_sizes'] += batch_size
        img_id += 1

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = net(z)
        fake_img[fake_img>1] = 1
        fake_img[fake_img<0] = 0

        psnr_t = psnr(fake_img.data.cpu(), real_img.data.cpu())
        psnrs += psnr_t

        g_loss = mse_loss(fake_img, real_img)

        running_results['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(desc='[%d] Loss_G: %.4f' % (
            epoch, running_results['g_loss'] / running_results['batch_sizes']))

        if opt.save_img > 0:
            res = fake_img.data.cpu()
            res = torch.squeeze(res, 0)
            res = ToPILImage()(res)
            res.save(save_dir + '/res_'+str(img_id)+'_'+str(psnr_t)+'.png')

    print("averate psnrs is: ", psnrs/img_id)
