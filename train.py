#!/usr/bin/env python

import argparse
import os

parser = argparse.ArgumentParser(description='Create a dataset from images')
parser.add_argument('datasetDirectory', type=unicode, help="the directory containing the dataset")
parser.add_argument('datasetName', type=unicode, help="the name of the dataset")
parser.add_argument('name', type=unicode, help="the name of the model")
parser.add_argument('--size', type=int, help="the size of the dictionary, must be a perfect square number (default 256)", default=256)
parser.add_argument('--overlap', type=int, help="the overlap size of the groups (default 4)", default=4)
parser.add_argument('--lambda1', type=float, help="the regularization parameter (default 0.2)", default=0.2)
parser.add_argument('--batchsize', type=int, help="the number of samples per minibatch (default 100)", default=100)
parser.add_argument('--device', type=str, help="the device to run the operation on (default /cpu:0)", default="/cpu:0")

parser.add_argument('--lassoIterations', type=int, help="the number of iterations for the lasso optimization (default 100)", default=100)
parser.add_argument('--out', type=unicode, help="the output directory", default=".")
parser.add_argument('--resume', type=bool, help="resume if model file exists (default yes)", default=True)
args = parser.parse_args()

from dictionary_learning.dataset import createDataset, Dataset
from dictionary_learning.utils import displayPatches, showArray
from dictionary_learning.learn import StructuredDictionaryLearner
import numpy as np
import tensorflow as tf
import math

ds = Dataset(name = args.datasetName, datasetDirectory = args.datasetDirectory)
ds.filterStddev(0.3)

sz = int(math.sqrt(args.size))
if args.size != sz * sz:
    print "Dictionary size must be a perfect square."
    exit(-1)

with tf.device(args.device):
    learner = StructuredDictionaryLearner(
        dictwidth = sz,
        patchsize = ds.patchSize * ds.patchSize * 3 if ds.color else ds.patchSize * ds.patchSize,
        batchsize = args.batchsize,
        lambda1 = args.lambda1,
        overlap = args.overlap,
        color = ds.color)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

learner.init(session)

try: os.makedirs(args.out)
except: pass

modelFile = os.path.join(args.out, args.name + ".pkl")
imageFile = os.path.join(args.out, args.name + ".png")

if args.resume:
    if os.path.exists(modelFile):
        with open(modelFile, "rb") as f:
            learner.loadModel(session, f)
else:
    if os.path.exists(modelFile):
        print "Model file exists."
        exit(-1)

while True:
    learner.train(session, ds, 100, lassoIterations = args.lassoIterations)

    learner.saveImage(session, imageFile)
    with open(modelFile + ".tmp", "wb") as f:
        learner.saveModel(session, f)

    os.rename(modelFile + ".tmp", modelFile)