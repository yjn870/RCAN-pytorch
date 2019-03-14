import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import RCAN

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    model = RCAN(opt)

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    filename = os.path.basename(opt.image_path).split('.')[0]

    input = pil_image.open(opt.image_path).convert('RGB')

    lr = input.resize((input.width // opt.scale, input.height // opt.scale), pil_image.BICUBIC)

    bicubic = lr.resize((input.width, input.height), pil_image.BICUBIC)
    bicubic.save(os.path.join(opt.outputs_dir, '{}_x{}_bicubic.png'.format(filename, opt.scale)))

    input = transforms.ToTensor()(lr).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input)

    output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    output = pil_image.fromarray(output, mode='RGB')
    output.save(os.path.join(opt.outputs_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))
