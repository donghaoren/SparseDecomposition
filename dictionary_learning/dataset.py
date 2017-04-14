from PIL import Image
from scipy import ndimage
import random
import numpy as np
import os
import json
from dictionary_learning.utils import displayPatches, showArray

def zcaWhitening(inputs):
    mean = np.mean(inputs, axis = 1)
    inputs = inputs - mean
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 0.01
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S + epsilon))), U.T)
    return mean, ZCAMatrix, np.dot(ZCAMatrix, inputs)   #Data whitening

def generatePatches(I, N, size, color):
    print "  Generating patches:", I.shape
    rsize = size * 4
    x0 = size
    y0 = size
    if color:
        for i in range(N):
            x = random.randint(0, I.shape[0] - rsize - 1)
            y = random.randint(0, I.shape[1] - rsize - 1)
            patch = I[x:x+rsize, y:y+rsize, :]
            angle = random.uniform(0, 360)
            patch = ndimage.rotate(patch, angle, reshape = False, mode = "nearest")
            patch = patch[x0:x0+size*2, y0:y0+size*2, :]
            patch = ndimage.zoom(patch, (0.5, 0.5, 1), mode = "nearest")
            # Centering
            patch = patch - np.mean(patch, axis = (0, 1))
            yield np.reshape(patch, (size * size * 3))
    else:
        for i in range(N):
            x = random.randint(0, I.shape[0] - rsize - 1)
            y = random.randint(0, I.shape[1] - rsize - 1)
            patch = I[x:x+rsize, y:y+rsize]
            angle = random.uniform(0, 360)
            patch = ndimage.rotate(patch, angle, reshape = False, mode = "nearest")
            patch = patch[x0:x0+size*2, y0:y0+size*2]
            patch = ndimage.zoom(patch, (0.5, 0.5), mode = "nearest")
            # Centering
            patch = patch - np.mean(patch)
            yield np.reshape(patch, (size * size))

def enumerateImages(directory, color):
    # Get training images
    for root, dirs, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(path)[1].lower()
            if ext == ".jpg" or ext == ".png":
                img = Image.open(path)
                I = np.array(img) / 255.0
                if color:
                    if len(I.shape) == 2:
                        I = np.dstack([ I, I, I ])
                else:
                    if len(I.shape) == 3:
                        I = I[:,:,0] * 0.2126 + I[:,:,1] * 0.7152 + I[:,:,2] * 0.0722

                yield path, I

def processImages(directory, patchSize, numPatches, color):
    result = []
    for path, I in enumerateImages(directory, color = color):
        print path, I.shape
        scale = 1600.0 / I.shape[0]
        if color:
            I = ndimage.zoom(I, [ scale, scale, 1 ], mode = "nearest")
        else:
            I = ndimage.zoom(I, [ scale, scale ], mode = "nearest")
        N = numPatches
        while I.shape[0] > 64:
            for patch in generatePatches(I, N, patchSize, color):
                result.append(patch)
            if color:
                I = ndimage.zoom(I, [ 0.5, 0.5, 1 ], mode = "nearest")
            else:
                I = ndimage.zoom(I, [ 0.5, 0.5 ], mode = "nearest")
            N = N / 4
    return np.matrix(result).T

class Dataset:
    def __init__(self, name, datasetDirectory = "."):
        with open(os.path.join(datasetDirectory, name + ".manifest.json"), "rb") as f:
            self.manifest = json.load(f)

        self.X = np.load(os.path.join(datasetDirectory, self.manifest["X"]))
        self.ZCAMatrix = np.load(os.path.join(datasetDirectory, self.manifest["ZCAMatrix"]))
        self.mean = np.load(os.path.join(datasetDirectory, self.manifest["mean"]))
        self.color = self.manifest["color"]
        self.patchSize = self.manifest["patchSize"]

    def filterStddev(self, stdMin):
        sd = np.std(self.X, axis = 0)
        ids = np.where(sd > stdMin)
        self.X = self.X.T[ids].T

    def getBatch(self, size):
        return self.X[:, np.random.randint(0, self.X.shape[1], size)]

    def probe(self, imageFile):
        showArray(displayPatches(self.getBatch(1024), color = self.color), imageFile)

def createDataset(imagesDirectory, name, datasetDirectory = ".", patchSize = 16, numPatchesPerImage = 20000, color = True):
    print "Loading images..."
    X = processImages(imagesDirectory, patchSize, numPatchesPerImage, color)
    print X.shape
    print "Whitening..."
    mean, matrix, X = zcaWhitening(X)

    print "Saving..."

    try: os.makedirs(datasetDirectory)
    except: pass

    np.save(os.path.join(datasetDirectory, name + ".mean.npy"), mean)
    np.save(os.path.join(datasetDirectory, name + ".matrix.npy"), matrix)
    np.save(os.path.join(datasetDirectory, name + ".X.npy"), X)
    with open(os.path.join(datasetDirectory, name + ".manifest.json"), "wb") as f:
        f.write(json.dumps({
            "name": name,
            "imagesDirectory": imagesDirectory,
            "patchSize": patchSize,
            "color": color,
            "X": name + ".X.npy",
            "ZCAMatrix": name + ".matrix.npy",
            "mean": name + ".mean.npy"
        }))