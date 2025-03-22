import argparse
parser = argparse.ArgumentParser()
parser.add_argument('teacher')
parser.add_argument('student')
parser.add_argument('undersample_class', type=int)
parser.add_argument('output')
parser.add_argument('--teacher-confidence', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--imsave', action='store_true')
parser.add_argument('--only-undersample', action='store_true')
args = parser.parse_args()

import mydatasets, mymodels
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d

(Xtr, Ytr), _ = mydatasets.mnist()
Xtr, Ytr = mydatasets.undersample(Xtr, Ytr, args.undersample_class)

X_over = Xtr[Ytr != args.undersample_class]
Y_over = Ytr[Ytr != args.undersample_class]
X_under = Xtr[Ytr == args.undersample_class]
Y_under = Ytr[Ytr == args.undersample_class]

teacher = mymodels.load(args.teacher)
student = mymodels.load(args.student)
student_input2latent = mymodels.subset(student, 'input', 'latent')
#student_input2latent.summary()
student_latent2image = mymodels.subset(student, 'latent', 'decoder')
#student_latent2image.summary()

# choose top-100 observations, the student is more unsure about
ix = student.predict(X_over)[0][:, args.undersample_class].argsort()[::-1]
ix = ix[:100]
wrongfreqs = np.bincount(Y_over[ix]) / len(ix)

# find a random undersample and do 25 interpolations of each
jx = np.random.choice(len(X_under), len(ix))
li = student_input2latent.predict(X_over[ix])
lj = student_input2latent.predict(X_under[jx])

X_new = []
for i in range(1, 21):  # 23 times
    coef = i/21
    l = li*(1-coef) + lj*coef
    x = student_latent2image(l)
    X_new += list(x)
X_new = np.array(X_new)

if args.imsave:
    from skimage.io import imsave
    def my_imsave(fname, x):
        p = tf.nn.softmax(teacher.predict(x[np.newaxis])[0])
        p_this = p[0, Y_over[ix[i]]]
        p_other = p[0, args.undersample_class]
        imsave(fname + f'-%d-%d.png' % (p_this*1000, p_other*1000), ((1-x[..., 0])*255).astype(np.uint8))
    for i in range(100):
        my_imsave(f'digits{args.undersample_class}-y{Y_over[ix[i]]}-{i}-0', X_over[ix[i]])
        for j in range(25):
            my_imsave(f'digits{args.undersample_class}-y{Y_over[ix[i]]}-{i}-{j+1}', X_new[i+j*100])
        my_imsave(f'digits{args.undersample_class}-y{Y_over[ix[i]]}-{i}-25', X_under[jx[i]])

# check which ones the teacher is sure is one way or the other
Yhat = tf.nn.softmax(teacher.predict(X_new)[0]).numpy()
if args.only_undersample:
    ix = np.logical_and(np.any(Yhat >= args.teacher_confidence, 1), Yhat.argmax(1) == args.undersample_class)
else:
    ix = np.any(Yhat >= args.teacher_confidence, 1)
X_new = X_new[ix]
Y_new = Yhat[ix].argmax(1)
print('%s,%d,%s' % (args.output, len(X_new), ','.join(map(str, wrongfreqs))))

# re-train
X = np.concatenate((Xtr, X_new))
Y = np.concatenate((Ytr, Y_new))
student.fit(X, (Y, X), 128, args.epochs, 0)
student.save(args.output)
