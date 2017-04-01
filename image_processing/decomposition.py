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

class Decomposition:
    def __init__(self, dictionary):
        self.dictionary = dictionary

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

        xySkip = 16
        xRange = range(0, I.shape[0] - self.dictionary.patchWH + 1, xySkip)
        yRange = range(0, I.shape[1] - self.dictionary.patchWH + 1, xySkip)
        N = len(xRange) * len(yRange)
        patches = np.zeros((self.dictionary.patchsize, N))
        xys = []
        means = []
        i = 0
        for x in xRange:
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

        Ir = [ np.zeros(I.shape, dtype = np.float32) for i in range(self.dictionary.size) ]
        Im = np.zeros(I.shape, dtype = np.float32)
        Ic = np.zeros(I.shape, dtype = np.float32)

        Dx = self.dictionary.ZD

        for i in range(N):
            i0 = result.indptr[i]
            i1 = result.indptr[i + 1]
            indices = result.indices[i0:i1]
            data = result.data[i0:i1]
            encoding = zip(indices.tolist(), data.tolist())
            x, y = xys[i]
            if self.dictionary.isColor:
                for idx, w in encoding:
                    Ir[idx][x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH,:] += (Dx[:, idx] * w).reshape((self.dictionary.patchWH, self.dictionary.patchWH, 3))
            else:
                for idx, w in encoding:
                    Ir[idx][x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += (Dx[:, idx] * w).reshape((self.dictionary.patchWH, self.dictionary.patchWH))
            Im[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += means[i]
            Ic[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += 1



        self.Ir = Ir
        self.Im = Im
        self.Ic = Ic

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

    def save(self, imageFile):
        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "wb") as f:
            pickle.dump([ self.Ir, self.Im, self.Ic ], f)

    def load(self, imageFile):
        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "rb") as f:
            Ir, Im, Ic = pickle.load(f)
        self.Ir = Ir
        self.Im = Im
        self.Ic = Ic

    def recompose(self, imageFile, outputFile, ids = None):
        Ir = self.Ir
        Im = self.Im
        Ic = self.Ic

        result = None

        if ids == None:
            I = Im
            for r in Ir:
                I = I + r
            result = I / Ic
        else:
            I = Im
            for r, idx in zip(Ir, range(len(Ir))):
                if idx in ids:
                    I = I + r
            result = I / Ic


        a = np.uint8(np.clip(result, 0, 1) * 255)
        Image.fromarray(a).save(self.getAuxFile(imageFile, outputFile), "png")

class HierarchicalDecomposition:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def getAuxFile(self, image, auxFile):
        d = os.path.dirname(image)
        withoutext = os.path.splitext(os.path.basename(image))[0]
        return os.path.join(d, withoutext + "." + auxFile)

    def encodePatches(self, I, lambda1, extractMean = True):
        xySkip = 1
        xRange = range(0, I.shape[0] - self.dictionary.patchWH + 1, xySkip)
        yRange = range(0, I.shape[1] - self.dictionary.patchWH + 1, xySkip)


        Ir = [ np.zeros(I.shape, dtype = np.float32) for i in range(self.dictionary.size) ]
        Im = np.zeros(I.shape, dtype = np.float32)
        Ic = np.zeros(I.shape, dtype = np.float32)

        Dx = self.dictionary.ZD

        for xs in splitArray(xRange, 100):
            N = len(yRange) * len(xs)
            patches = np.zeros((self.dictionary.patchsize, N))
            xys = []
            means = []
            i = 0
            for x in xs:
                for y in yRange:
                    xys.append([ x, y ])
                    patch = I[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH]
                    if self.dictionary.isColor:
                        mean = np.mean(patch, axis = (0, 1))
                    else:
                        mean = np.mean(patch)
                    if not extractMean:
                        mean *= 0
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
                if self.dictionary.isColor:
                    for idx, w in encoding:
                        Ir[idx][x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH,:] += (Dx[:, idx] * w).reshape((self.dictionary.patchWH, self.dictionary.patchWH, 3))
                else:
                    for idx, w in encoding:
                        Ir[idx][x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += (Dx[:, idx] * w).reshape((self.dictionary.patchWH, self.dictionary.patchWH))
                Im[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += means[i]
                Ic[x:x+self.dictionary.patchWH,y:y+self.dictionary.patchWH] += 1
        print Im[:,:,0], Ic[:,:,0]
        return Ir, Im, Ic

    def preprocess(self, imageFile, lambda1 = 0.05):
        # Get all patches from image
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
        while min(Ic.shape[0], Ic.shape[1]) >= self.dictionary.patchWH:
            hierarchy = [ Ic ] + hierarchy
            Ic = zoomImage(Ic, 0.5)

        # Encode hierarchy:
        imgPrevious = None
        layerIndex = 1
        result = []
        for img in hierarchy:
            if imgPrevious is not None:
                offset = zoomImage(imgPrevious, 2)
                Ir, Im, Ic = self.encodePatches(img - offset, lambda1)
            else:
                offset = None
                Ir, Im, Ic = self.encodePatches(img, lambda1)

            Ireconstruct = (sumArrays(Ir) + Im) / np.maximum(Ic, 0.00001)
            if offset is not None:
                Ireconstruct += offset

            result.append(( Ir, Im, Ic ))

            saveImage(Ireconstruct, self.getAuxFile(imageFile, "layer-%d.png" % layerIndex), scale = True)

            imgPrevious = Ireconstruct

            layerIndex += 1

        # Reconstruct images
        self.encoding = result

    def save(self, imageFile):
        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "wb") as f:
            pickle.dump(self.encoding, f)

    def load(self, imageFile):
        with open(self.getAuxFile(imageFile, "decomposition.pkl"), "rb") as f:
            encoding = pickle.load(f)
        self.encoding = encoding

    def recompose(self, imageFile, outputFile, ids = None):
        if ids == None:
            ids = set(range(self.dictionary.size))

        imgPrevious = None
        for Ir, Im, Ic in self.encoding:
            I = Im

            for r, idx in zip(Ir, range(len(Ir))):
                if idx in ids:
                    I = I + r
            I = I / np.maximum(Ic, 0.00001)
            if imgPrevious is not None:
                I = I + zoomImage(imgPrevious, 2)

            imgPrevious = I

        saveImage(I, self.getAuxFile(imageFile, outputFile))

    def interactive(self, imageFile, port = 8888):
        import tornado.ioloop
        import tornado.web
        import json

        dictionary = self.dictionary
        myself = self

        class ComposeHandler(tornado.web.RequestHandler):
            def get(self):
                ids = map(int, filter(lambda x: x != "", self.get_argument("ids", default = "").split(",")))
                myself.recompose(imageFile, "server.png", ids)
                with open(myself.getAuxFile(imageFile, "server.png"), "rb") as f:
                    self.add_header("content-type", "image/png")
                    self.write(f.read())

        class DictionaryHandler(tornado.web.RequestHandler):
            def get(self):
                self.add_header("content-type", "application/json")
                self.write(json.dumps({
                    "patchWH": dictionary.patchWH,
                    "D": dictionary.D.T.tolist(),
                    "ZD": dictionary.ZD.T.tolist(),
                    "size": dictionary.size,
                    "dictwidth": dictionary.dictwidth
                }))

        app = tornado.web.Application([
            (r"/api/compose", ComposeHandler),
            (r"/api/dictionary", DictionaryHandler),
            (r"/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")})
        ])

        app.listen(port)
        print "Listening on http port 8888..."
        tornado.ioloop.IOLoop.current().start()