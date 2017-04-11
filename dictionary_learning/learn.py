import tensorflow as tf
import numpy as np
import time
import pickle
from utils import showArray, displayPatches

def createStructuredDictionary(width, patchsize, overlap):
    count = width * width
    D0 = np.random.uniform(-0.01, 0.01, (patchsize, count))
    DG = np.zeros((count, count))
    for x in range(width):
        for y in range(width):
            idx = y * width + x
            ixs = [(x + i) % width for i in range(overlap)]
            iys = [(y + i) % width for i in range(overlap)]
            for ix in ixs:
                for iy in iys:
                    i = iy * width + ix
                    DG[idx, i] = 1
    return D0, DG

class StructuredDictionaryLearner:

    def __init__(self, dictwidth=20, patchsize=1024, batchsize=100, lambda1=0.2, overlap = 4, color = False, structure = None):
        if structure:
            D0, DG = structure
            dictsize = D0.shape[1]
            patchsize = D0.shape[0]
            self.dictsize = dictsize
            self.patchsize = patchsize
            self.batchsize = batchsize
            self.lambda1 = lambda1
            self.color = color
        else:
            dictsize = dictwidth * dictwidth
            self.dictsize = dictsize
            self.patchsize = patchsize
            self.batchsize = batchsize
            self.lambda1 = lambda1
            self.color = color

            D0, DG = createStructuredDictionary(dictwidth, patchsize, overlap)

        batch = tf.placeholder(dtype=tf.float32)
        j = tf.placeholder(dtype=tf.int32)

        D = tf.get_variable("dictionary", [patchsize, dictsize], tf.float32, tf.random_normal_initializer(0, 0.01))
        X = tf.get_variable("input", [patchsize, batchsize], tf.float32, tf.constant_initializer(0.0))
        alpha = tf.get_variable("alpha", [dictsize, batchsize], tf.float32, tf.constant_initializer(0.01))
        c_lambda1 = tf.constant(lambda1, dtype=tf.float32)
        c_DG = tf.constant(DG, dtype=tf.float32)

        assignX = X.assign(batch)

        base_loss = tf.reduce_sum(tf.square(tf.matmul(D, alpha) - X)) / 2.0
        lasso_loss = tf.reduce_sum(tf.abs(alpha)) * c_lambda1
        lasso_group_loss = tf.reduce_sum(
            tf.sqrt(tf.matmul(c_DG, tf.square(alpha)) + 1e-6)) * c_lambda1

        loss = base_loss + lasso_group_loss

        beta = tf.placeholder(dtype=tf.float32)
        A = tf.get_variable("A", [dictsize, dictsize], tf.float32, tf.constant_initializer(0))
        Anew = A * beta + tf.matmul(alpha, tf.transpose(alpha))
        B = tf.get_variable("B", [patchsize, dictsize], tf.float32, tf.constant_initializer(0))
        Bnew = B * beta + tf.matmul(X, tf.transpose(alpha))

        reset_alpha = alpha.assign(0.01 * np.ones((dictsize, batchsize)))

        # Optimize the LASSO function
        optimizer = tf.train.AdamOptimizer(learning_rate=0.5)
        do_lasso_step = optimizer.minimize(loss, var_list=[alpha])
        do_init_lasso = tf.variables_initializer(filter(lambda x: x != None, [optimizer.get_slot(alpha, name) for name in optimizer.get_slot_names()]))

        assignAB = [A.assign(Anew), B.assign(Bnew)]

        uj = (B[:, j:j + 1] - tf.matmul(D, A[:, j:j + 1])) / A[j, j] + D[:, j:j + 1]
        dj = uj / tf.maximum(tf.sqrt(tf.reduce_sum(uj * uj)), 1)
        assignDj = D[:, j:j + 1].assign(dj)

        self.D = D
        self.A = A
        self.B = B
        self.assignX = assignX
        self.loss = loss
        self.reset_alpha = reset_alpha
        self.do_lasso_step = do_lasso_step
        self.do_init_lasso = do_init_lasso
        self.assignAB = assignAB
        self.assignDj = assignDj
        self.beta = beta
        self.j = j
        self.batch = batch

        self.t = 1

    def init(self, session):
        session.run(tf.global_variables_initializer())

    def train(self, session, dataset, iterations=100, lassoIterations=100):
        for iteration in range(iterations):
            # TT = TimeMeasure()

            session.run(self.assignX, {self.batch: dataset.getBatch(self.batchsize)})

            # TT.show("assignX")

            session.run(self.reset_alpha)
            session.run(self.do_init_lasso)
            for i in range(lassoIterations):
                session.run(self.do_lasso_step)

            # TT.show("lasso")

            _theta = self.t * self.batchsize if self.t < self.batchsize else self.batchsize ** 2 + self.t - self.batchsize
            _beta = (_theta + 1.0 - self.batchsize) / (_theta + 1.0)

            session.run(self.assignAB, {self.beta: _beta})

            # TT.show("assignAB")

            for _j in range(self.dictsize):
                session.run(self.assignDj, {self.j: _j})
            # TT.show("updateD")

            print "Iteration %06d" % self.t
            # TT.show("assignD")
            self.t = self.t + 1

    def saveModel(self, session, file):
        pickle.dump([
            self.t,
            session.run(self.D),
            session.run(self.A),
            session.run(self.B)
        ], file)

    def loadModel(self, session, file, loadAB = True):
        t, D, A, B = pickle.load(file)
        self.t = t
        session.run(self.D.assign(D))
        if loadAB:
            session.run(self.A.assign(A))
            session.run(self.B.assign(B))

    def getD(self, session):
        return session.run(self.D)

    def saveImage(self, session, f):
        return showArray(displayPatches(self.getD(session), color = self.color), f)
