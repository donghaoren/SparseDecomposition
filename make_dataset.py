#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Create a dataset from images')
parser.add_argument('imagesDirectory', type=unicode, help="the directory containing the images")
parser.add_argument('name', type=unicode, help="the name of the dataset")
parser.add_argument('--out', type=unicode, help="the output directory", default=".")
parser.add_argument('--patchsize', type=int, help="the width/height of the patches", default=16)
parser.add_argument('--count', type=int, help="number of patches per image", default=20000)
parser.add_argument('--color', type=bool, help="generate color patches", default=False)
args = parser.parse_args()

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
