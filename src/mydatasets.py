import tensorflow as tf
import numpy as np

def undersample(X, Y, undersample_class):
    rand = np.random.RandomState(123)
    X1 = X[Y != undersample_class]
    Y1 = Y[Y != undersample_class]
    X2 = X[Y == undersample_class]
    Y2 = Y[Y == undersample_class]
    ix = rand.choice(len(X2), 10, False)
    X2 = X2[ix]
    Y2 = Y2[ix]
    return np.concatenate((X1, X2)), np.concatenate((Y1, Y2))

def mnist():
    (Xtr, Ytr), (Xts, Yts) = tf.keras.datasets.mnist.load_data()
    pad = ((0, 0), (2, 2), (2, 2))  # padding: 28x28 -> 32x32
    Xtr = np.pad((Xtr/255).astype(np.float32), pad)[..., np.newaxis]
    Xts = np.pad((Xts/255).astype(np.float32), pad)[..., np.newaxis]
    Ytr = Ytr.astype(np.int32)
    Yts = Yts.astype(np.int32)
    return (Xtr, Ytr), (Xts, Yts)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    (Xtr, Ytr), (Xts, Yts) = mnist()
    print('train:', Xtr.shape, Xtr.dtype, Ytr.shape, Ytr.dtype)
    print('test: ', Xts.shape, Xts.dtype, Yts.shape, Yts.dtype)
    plt.imshow(Xtr[0], cmap='gray_r')
    plt.show()
