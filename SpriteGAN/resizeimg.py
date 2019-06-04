from __future__ import division
from PIL import Image
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser(description='SpriteGAN')
parser.add_argument('--path_to_dataset', type=str, default='.', help='Path to dataset')
parser.add_argument('--path_to_output_dir', type=str, default='.', help='Path to output directory')
args = parser.parse_args()

# Preprocess images
# Resize to 32x32x3 pixels

def resize_img(img, loc):
    ww = 32
    size = (ww, ww)
    new_img = img.split("/")[-1]
    fn = loc + new_img
    fn_flipped = loc + 'flipped' + new_img
    img = Image.open(img)
    img = img.resize(size)
    flipp_img = img.copy()
    flipped_img = flipp_img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(fn)
    flipped_img.save(fn_flipped)

path_to_img = args.path_to_dataset
path_to_output_dir = args.path_to_output_dir
original = []
# Get file names
for filename in glob.glob(path_to_img+'/*.gif'):
    original.append(filename)

# Resize images and save them
t = 0
for image in original:
    resize_img(image, path_to_output_dir)

