#!/usr/bin/env python

import argparse
import os

parser = argparse.ArgumentParser(description='Run image denoising with learnt dictionary')
parser.add_argument('dictionary', type=unicode, help="the directory containing the dataset")
parser.add_argument('zcaMatrix', type=unicode, help="the name of the dataset")
parser.add_argument('image', type=unicode, help="the name of the model")
parser.add_argument('--lambda1', type=float, help="the regularization parameter (default 0.2)", default=0.2)

args = parser.parse_args()

from image_processing.denoising import Denoising
from dictionary_learning.dictionary import Dictionary

dictionary = Dictionary(args.dictionary, args.zcaMatrix)

imageFile = args.image

decomp = Denoising(dictionary)

decomp.denoising(imageFile, "denoise.png", lambda1 = args.lambda1)