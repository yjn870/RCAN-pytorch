import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import RCAN
from dataset import Dataset
from utils import AverageMeter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    model = RCAN(opt).to(device)
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size, opt.scale, opt.use_fast_loader)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
