#!/usr/bin/env python

import argparse

def str2bool(v):
  return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Create a dataset from images')
parser.register('type','bool',str2bool) # add type keyword to registries

parser.add_argument('imagesDirectory', type=unicode, help="the directory containing the images")
parser.add_argument('name', type=unicode, help="the name of the dataset")
parser.add_argument('--out', type=unicode, help="the output directory (default: .)", default=".")
parser.add_argument('--patchsize', type=int, help="the width/height of the patches (default: 16)", default=16)
parser.add_argument('--count', type=int, help="number of patches per image (default 20000)", default=20000)
parser.add_argument('--color', type='bool', help="generate color patches (default no)", default=False)
args = parser.parse_args()

print args
exit(0)

from dictionary_learning.dataset import createDataset, Dataset
from dictionary_learning.utils import displayPatches, showArray
import numpy as np

createDataset(
    imagesDirectory = args.imagesDirectory,
    name = args.name,
    datasetDirectory = args.out,
    patchSize = args.patchsize,
    numPatchesPerImage = args.count,
    color = args.color
)
