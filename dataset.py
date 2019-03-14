import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            hr = tf.read_file(self.image_files[idx])
            hr = tf.image.decode_jpeg(hr, channels=3)
            hr = pil_image.fromarray(hr.numpy())
        else:
            hr = pil_image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # degrade lr with Bicubic
        lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        # normalization
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    def __len__(self):
        return len(self.image_files)
