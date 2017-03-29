import spams
import pickle
import os
import numpy as np
from PIL import Image

class Decomposition:
    def __init__(self, dictionary, matrix = None):
        self.dictionary = dictionary
        if matrix != None:
            self.ZCAMatrix = np.load(matrix)
            self.ZCAMatrixInverse = np.linalg.inv(self.ZCAMatrix)
        else:
            self.ZCAMatrix = None

    def getAuxFile(self, image, auxFile):
        d = os.path.dirname(image)
        withoutext = os.path.splitext(os.path.basename(image))[0]
        return os.path.join(d, withoutext + "." + auxFile)

    def preprocess(self, imageFile, lambda1 = 0.1):
        # Get all patches from image
        img = Image.open(imageFile)
        I = np.array(img) / 255.0
        if self.dictionary.isColor:
            if len(I.shape) == 2:
                I = np.dstack([ I, I, I ])
        else:
            if len(I.shape) == 3:
                I = I[:,:,0] * 0.2126 + I[:,:,1] * 0.7152 + I[:,:,2] * 0.0722

        xCount = I.shape[0] - self.dictionary.patchWH + 1
        yCount = I.shape[1] - self.dictionary.patchWH + 1
        N = xCount * yCount
        patches = np.zeros((self.dictionary.patchsize, N))
        xys = []
        means = []
        for x in range(xCount):
            for y in range(yCount):
                i = x * yCount + y
                xys.append([ x, y ])
                patch = I[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH]
                mean = np.mean(patch)
                patch = patch - mean
                means.append(mean)
                patches[:,i] = np.reshape(patch, self.dictionary.patchsize)

        if self.ZCAMatrix is not None:
            patches = np.dot(self.ZCAMatrix, patches)

        patches = np.asfortranarray(patches, dtype = np.float64)

        # Decompose
        result = spams.lasso(
            patches,
            D = np.asfortranarray(self.dictionary.D, dtype = np.float64),
            return_reg_path = False, mode = 2, numThreads= -1,
            lambda1 = lambda1
        )

        Ir = [ np.zeros(I.shape, dtype = np.float64) for i in range(self.dictionary.size) ]
        Im = np.zeros(I.shape, dtype = np.float64)
        Ic = np.zeros(I.shape, dtype = np.float64)

        Dx = self.dictionary.D
        if self.ZCAMatrix is not None:
            Dx = np.dot(self.ZCAMatrixInverse, self.dictionary.D)

        for i in range(N):
            i0 = result.indptr[i]
            i1 = result.indptr[i + 1]
            indices = result.indices[i0:i1]
            data = result.data[i0:i1]
            encoding = zip(indices.tolist(), data.tolist())
            x, y = xys[i]
            for idx, w in encoding:
                Ir[idx][x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += (Dx[:, idx] * w).reshape((self.dictionary.patchWH, self.dictionary.patchWH))
            Im[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += means[i]
            Ic[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += 1

        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "wb") as f:
            pickle.dump([ Ir, Im, Ic ], f)

        # decomposition = []

        # for i in range(N):
        #     i0 = result.indptr[i]
        #     i1 = result.indptr[i + 1]
        #     indices = result.indices[i0:i1]
        #     data = result.data[i0:i1]
        #     encoding = zip(indices.tolist(), data.tolist())
        #     encoding = sorted(encoding, key = lambda x: abs(x[1]), reverse = True)

        #     decomposition.append([ xys[i], float(means[i]), encoding ])

        # # Save result
        # with open(self.getAuxFile(imageFile, "decomposition.pkl"), "wb") as f:
        #     pickle.dump([ I.shape, decomposition ], f)

    def recompose(self, imageFile, outputFile, ids = None):
        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "rb") as f:
            Ir, Im, Ic = pickle.load(f)

        if ids == None:
            I = Im
            for r in Ir:
                I += r
            result = I / Ic
        else:
            I = np.zeros(Im.shape)
            for r, idx in zip(Ir, range(len(Ir))):
                if idx in ids:
                    I += r
            result = I / Ic + 0.5


        a = np.uint8(np.clip(result, 0, 1) * 255)
        Image.fromarray(a).save(self.getAuxFile(imageFile, outputFile), "png")