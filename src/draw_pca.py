import argparse
parser = argparse.ArgumentParser()
parser.add_argument('student')
parser.add_argument('undersample_class', type=int)
args = parser.parse_args()

import mydatasets, mymodels
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(8, 3.5))

(Xtr, Ytr), _ = mydatasets.mnist()

student = mymodels.load(args.student)
student_input2latent = mymodels.subset(student, 'input', 'latent')
student_input2latent.summary()

L = student_input2latent.predict(Xtr)
L_min = L.min(0)[np.newaxis]
L_max = L.max(0)[np.newaxis]
L = (L-L_min)/(L_max-L_min)  # normalize

pca = PCA(2)
L2 = pca.fit_transform(L.reshape((len(L), -1)))

for k in range(10):
    if k == args.undersample_class: continue
    lk = L2[Ytr == k].mean(0)
    lu = L2[Ytr == args.undersample_class].mean(0)
    plt.plot([lk[0], lu[0]], [lk[1], lu[1]], linestyle='--', color='gray', linewidth=0.5)

cmap = matplotlib.cm.get_cmap('hsv')
ix = np.linspace(0, len(L2), 250, False, dtype=int)
for l2, y in zip(L2[ix], Ytr[ix]):
    c = cmap(y/10) if y != args.undersample_class else 'black'
    plt.text(l2[0], l2[1], str(y), color=c, clip_on=True)

plt.xlim(np.percentile(L2[:, 0], 1), np.percentile(L2[:, 0], 99))
plt.ylim(np.percentile(L2[:, 1], 1), np.percentile(L2[:, 1], 99))

plt.margins(0, 0)
plt.savefig(f'pca{args.undersample_class}.pdf', bbox_inches='tight', pad_inches=0)

'''
X_under = Xtr[Ytr == args.undersample_class][:1]
lj = student_input2latent.predict(X_under)

for k in range(10):
    if k == args.undersample_class: continue
    Lk = L2
    Y = Ytr[Ytr == k][:1]
    li = student_input2latent.predict(X)

    L2 = []
    for i in range(50+1):  # 23 times
        coef = i/50
        l = li*(1-coef) + lj*coef
        l = (l-L_min)/(L_max-L_min)
        l2 = pca.transform(l.reshape(1, -1))
        L2 += list(l2)
    L2 = np.array(L2)
    plt.plot(L2[:, 0], L2[:, 1], linestyle='--', color='k')
'''
