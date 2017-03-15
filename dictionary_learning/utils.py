import numpy as np
import time

def displayPatchesGrayscale(D, scale=1.5):
    border = 1
    count = D.shape[1]
    patchsize = D.shape[0]
    wh = int(np.sqrt(count))
    patchwh = int(np.sqrt(patchsize))
    result = np.zeros(((patchwh + border) * wh + border,
                       (patchwh + border) * wh + border))
    maxABS = np.max(np.abs(D))
    D = D / maxABS / 2 * scale + 0.5
    for x in range(wh):
        for y in range(wh):
            patch = D[:, y * wh + x].reshape((patchwh, patchwh)).T
            result[1 + y * (patchwh + 1):1 + y * (patchwh + 1) + patchwh,
                   1 + x * (patchwh + 1):1 + x * (patchwh + 1) + patchwh] = patch
    return result

def displayPatchesColor(D, scale=1.5):
    border = 1
    count = D.shape[1]
    patchsize = D.shape[0] / 3
    wh = int(np.sqrt(count))
    patchwh = int(np.sqrt(patchsize))
    result = np.zeros(((patchwh + border) * wh + border,
                       (patchwh + border) * wh + border, 3))
    maxABS = np.max(np.abs(D))
    D = D / maxABS / 2 * scale + 0.5
    for x in range(wh):
        for y in range(wh):
            patch = D[:, y * wh + x].reshape((patchwh, patchwh, 3))
            result[1 + y * (patchwh + 1):1 + y * (patchwh + 1) + patchwh,
                   1 + x * (patchwh + 1):1 + x * (patchwh + 1) + patchwh] = patch
    return result

def displayPatches(D, color = False, scale = 1.5):
    if color:
        return displayPatchesColor(D, scale)
    else:
        return displayPatchesGrayscale(D, scale)

def showArray(a, f, fmt='png'):
    import PIL.Image
    a = np.uint8(np.clip(a, 0, 1) * 255)
    PIL.Image.fromarray(a).save(f, fmt)

class TimeMeasure:
    def __init__(self):
        self.t = time.time()

    def show(self, msg):
        t = time.time()
        print("%s: %.3fs" % (msg, t - self.t))
        self.t = t