import spams
import pickle
import os
import numpy as np
from PIL import Image

from scipy import ndimage

def zoomImage(I, factor):
    if len(I.shape) == 2:
        return ndimage.zoom(I, (factor, factor), mode = "nearest")
    else:
        return ndimage.zoom(I, (factor, factor, 1), mode = "nearest")

def zoomImageToSize(I, size):
    if len(I.shape) == 2:
        return ndimage.zoom(I, (float(size[0]) / I.shape[0], float(size[1]) / I.shape[1]), mode = "nearest")
    else:
        return ndimage.zoom(I, (float(size[0]) / I.shape[0], float(size[1]) / I.shape[1], 1), mode = "nearest")


def sumArrays(Is):
    I0 = Is[0]
    for i in Is[1:]:
        I0 = I0 + i
    return I0

def saveImage(I, file, scale = False):
    if scale:
        I = (I - np.min(I)) / (np.max(I) - np.min(I))
    a = np.uint8(np.clip(I, 0, 1) * 255)
    Image.fromarray(a).save(file, "png")

def splitArray(array, chunkSize):
    a = []
    for item in array:
        a.append(item)
        if len(a) >= chunkSize:
            yield a
            a = []
    if len(a) > 0:
        yield a

class Denoising:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def getAuxFile(self, image, auxFile):
        d = os.path.dirname(image)
        withoutext = os.path.splitext(os.path.basename(image))[0]
        return os.path.join(d, withoutext + "." + auxFile)

    def encodePatches(self, I, lambda1 = 0.1):
        if self.dictionary.isColor:
            if len(I.shape) == 2:
                I = np.dstack([ I, I, I ])
        else:
            if len(I.shape) == 3:
                I = I[:,:,0] * 0.2126 + I[:,:,1] * 0.7152 + I[:,:,2] * 0.0722

        Ir = np.zeros(I.shape, dtype = np.float32)
        Ic = np.zeros(I.shape, dtype = np.float32)

        Dx = self.dictionary.ZD

        xySkip = 1
        xRange = range(0, I.shape[0] - self.dictionary.patchWH + 1, xySkip)
        yRange = range(0, I.shape[1] - self.dictionary.patchWH + 1, xySkip)

        for xs in splitArray(xRange, 100):
            N = len(yRange) * len(xs)
            i = 0
            patches = np.zeros((self.dictionary.patchsize, N))
            xys = []
            means = []
            for x in xs:
                for y in yRange:
                    xys.append([ x, y ])
                    patch = I[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH]
                    if self.dictionary.isColor:
                        mean = np.mean(patch, axis = (0, 1))
                    else:
                        mean = np.mean(patch)
                    patch = patch - mean
                    means.append(mean)
                    patches[:,i] = np.reshape(patch, self.dictionary.patchsize)
                    i += 1

            if self.dictionary.ZCAMatrix is not None:
                patches = np.dot(self.dictionary.ZCAMatrix, patches)

            patches = np.asfortranarray(patches, dtype = np.float64)

            # Decompose
            result = spams.lasso(
                patches,
                D = np.asfortranarray(self.dictionary.D, dtype = np.float64),
                return_reg_path = False, mode = 2, numThreads= -1,
                lambda1 = lambda1
            )

            for i in range(N):
                i0 = result.indptr[i]
                i1 = result.indptr[i + 1]
                indices = result.indices[i0:i1]
                data = result.data[i0:i1]
                encoding = zip(indices.tolist(), data.tolist())
                x, y = xys[i]
                patch = np.zeros(Dx[:, 0].shape, np.float32)
                for idx, w in encoding:
                    patch += Dx[:, idx] * w
                if self.dictionary.isColor:
                    Ir[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH,:] += patch.reshape((self.dictionary.patchWH, self.dictionary.patchWH, 3))
                else:
                    Ir[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += patch.reshape((self.dictionary.patchWH, self.dictionary.patchWH))
                Ir[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += means[i]
                Ic[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += 1

        return Ir, Ic

    def denoising(self, imageFile, outputFile, lambda1 = 0.1):
        img = Image.open(imageFile)
        I = np.array(img) / 255.0
        if self.dictionary.isColor:
            if len(I.shape) == 2:
                I = np.dstack([ I, I, I ])
        else:
            if len(I.shape) == 3:
                I = I[:,:,0] * 0.2126 + I[:,:,1] * 0.7152 + I[:,:,2] * 0.0722

        # Create image hierarchy
        Ic = I
        hierarchy = []
        sizes = []
        while min(Ic.shape[0], Ic.shape[1]) >= self.dictionary.patchWH:
            hierarchy = [ Ic ] + hierarchy
            Ic = zoomImage(Ic, 0.5)

        # Encode hierarchy:
        imgPrevious = None
        layerIndex = 1
        for img in hierarchy:
            print "layer %d, %s" % (layerIndex, img.shape)
            if imgPrevious is not None:
                offset = zoomImageToSize(imgPrevious, img.shape[0:2])
                Ir, Ic = self.encodePatches(img - offset, lambda1)
            else:
                offset = None
                Ir, Ic = self.encodePatches(img, lambda1)

            Ireconstruct = (Ir) / np.maximum(Ic, 0.00001)
            if offset is not None:
                Ireconstruct += offset

            saveImage(Ireconstruct, self.getAuxFile(imageFile, "layer-%d.png" % layerIndex))

            imgPrevious = Ireconstruct

            layerIndex += 1

        saveImage(Ireconstruct, self.getAuxFile(imageFile, outputFile))
