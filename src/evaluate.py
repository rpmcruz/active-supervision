import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

import mydatasets, mymodels
import numpy as np
from sklearn import metrics
import pandas as pd

_, (Xts, Yts) = mydatasets.mnist()

model = mymodels.load(args.model)
Yhat = model.predict(Xts)[0].argmax(1)

scores = {
    'model': args.model,
    'acc': metrics.accuracy_score(Yts, Yhat),
    **{f'acc{k}': np.sum(np.logical_and(Yts == k, Yhat == k)) / np.sum(Yts == k) for k in range(Yts.max()+1)},
    'bacc': metrics.balanced_accuracy_score(Yts, Yhat),
}

df = pd.DataFrame([scores.values()], columns=scores.keys())
print(df.to_csv(index=False, header=False), end='')
