#!/usr/bin/env python

import argparse
import os
import numpy as np

def generateTonnetzStructure(patchsize):
    N = 84

    def spelling(x):
        return x % N

    def triangleA(start):
        return tuple(sorted(map(spelling, [ start, start + 7, start + 4 ])))
    def triangleB(start):
        return tuple(sorted(map(spelling, [ start, start + 7, start + 3 ])))

    def allTriangles():
        for s in range(N):
            t1 = triangleA(s)
            t2 = triangleB(s)
            yield t1
            yield t2

    c = dict()
    for i in range(N):
        c[i] = set()

    for triangle in set(allTriangles()):
        c[triangle[0]].add(triangle[1])
        c[triangle[0]].add(triangle[2])
        c[triangle[1]].add(triangle[0])
        c[triangle[1]].add(triangle[2])
        c[triangle[2]].add(triangle[0])
        c[triangle[2]].add(triangle[1])

    groups = set()

    for note in c:
        groups.add(tuple(sorted([ note ] + list(c[note]))))

    repeat = 5
    D0 = np.random.uniform(-0.01, 0.01, (patchsize, N * repeat))
    DG = np.zeros((len(groups) * repeat, N * repeat))
    idx = 0
    for g in groups:
        for i in range(repeat):
            for j in range(3):
                d = (i + j) % repeat
                for y in g:
                    DG[idx, y + d * N] = 1
            idx += 1
    return D0, DG

parser = argparse.ArgumentParser(description='Create a dataset from images')
parser.add_argument('datasetDirectory', type=unicode, help="the directory containing the dataset")
parser.add_argument('datasetName', type=unicode, help="the name of the dataset")
parser.add_argument('name', type=unicode, help="the name of the model")
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

patchsize = ds.patchSize * ds.patchSize * 3 if ds.color else ds.patchSize * ds.patchSize

with tf.device(args.device):
    learner = StructuredDictionaryLearner(
        structure = generateTonnetzStructure(patchsize),
        patchsize = patchsize,
        batchsize = args.batchsize,
        lambda1 = args.lambda1,
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
