import argparse
parser = argparse.ArgumentParser()
parser.add_argument('type', choices=['teacher', 'student'])
parser.add_argument('undersample_class', type=int)
parser.add_argument('output')
parser.add_argument('--epochs', type=int , default=50)
args = parser.parse_args()

import mydatasets, mymodels
import tensorflow as tf
import numpy as np

(Xtr, Ytr), _ = mydatasets.mnist()
if args.type == 'student':
    Xtr, Ytr = mydatasets.undersample(Xtr, Ytr, args.undersample_class)

model = mymodels.class_and_decoder(Xtr.shape[1:], Ytr.max()+1)
model.summary()

ce = tf.keras.losses.SparseCategoricalCrossentropy(True)
model.compile('adam', [ce, 'mse'])

model.fit(Xtr, (Ytr, Xtr), 128, args.epochs, 2)
model.save(args.output)
