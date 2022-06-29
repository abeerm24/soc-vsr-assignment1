# This code allows you to view files using a uniform image renderer
# Command to run code:
# python viewer.py --img_file "path-to-image"

import h5py
import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_file', type=str, required=True)
args = parser.parse_args()

cv2.imshow(args.img_file)
cv2.waitKey(0)