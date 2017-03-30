import math
import pickle
import numpy as np

class Dictionary:
    def __init__(self, file):
        with open(file, "rb") as f:
            D = np.load(f)

        self.D = D
        self.size = D.shape[1]
        self.patchsize = D.shape[0]
        self.dictwidth = int(math.sqrt(self.size))

        self.isColor = False
        self.patchWH = int(math.sqrt(self.patchsize))
        if self.patchsize % 3 == 0:
            candidateWH = int(math.sqrt(self.patchsize / 3))
            if candidateWH ** 2 * 3 == self.patchsize:
                self.isColor = True
                self.patchWH = candidateWH
