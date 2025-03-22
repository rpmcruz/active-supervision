import pandas as pd
import numpy as np

df = pd.read_csv('models/results-accuracy.csv')

# ignore teacher
df = df.iloc[1:]

# more columns
df['class'] = [int(s.split('-')[0][-1]) for s in df['model']]
df['iter'] = [int(s.split('-')[1][-1]) for s in df['model']]
df['rep'] = [int(s.split('-')[2][-1]) for s in df['model']]
df['acc_class'] = [row['acc%d' % row['class']] for _, row in df.iterrows()]
df['acc_other'] = [row[['acc%d' % k for k in range(10) if k != row['class']]].mean() for _, row in df.iterrows()]

# do average and deviation of reps
res = df.groupby(['class', 'iter']).agg({'class': 'first', 'iter': 'first', 'acc_class': ['mean', 'std'], 'acc_other': ['mean', 'std']})
res.columns = [col[0] if col[1] == 'first' else '_'.join(col).strip() for col in res.columns.values]
res.to_csv('results_acc_per_class.csv', index=False)

if False:
    # generate confidence tables for confidences

    print()
    print()

    tex = r'''\documentclass{standalone}
    \begin{document}
    \begin{tabular}{l|r|r|r}
    & \multicolumn{1}{c|}{50} & \multicolumn{1}{c|}{75} & \multicolumn{1}{c}{90} \\
    \hline
    '''
    ress = [pd.read_csv('results_acc_per_class-confidence%d.csv' % c) for c in (50, 75, 90)]
    for klass in (2, 5, 8):
        ress_klass = [r[r['class'] == klass] for r in ress]
        # https://www.researchgate.net/post/How_do_you_calculate_the_standard_deviations_when_subtracting_two_means_with_different_standard_deviations
        ress_gains = [100*(float(r[r['iter'] == 1]['acc_class_mean']) - float(r[r['iter'] == 0]['acc_class_mean'])) for r in ress_klass]
        ress_devs = [100*np.sqrt(float(r[r['iter'] == 1]['acc_class_std'])**2 + float(r[r['iter'] == 0]['acc_class_std'])**2) for r in ress_klass]
        ress_gainsdevs = []
        for g, d in zip(ress_gains, ress_devs):
            ress_gainsdevs.append(g)
            ress_gainsdevs.append(d)
        tex += '$k=%d$ & $%.1f\pm%04.1f$ & $%.1f\pm%04.1f$ & $%.1f\pm%04.1f$ \\\\\n' % (klass, *ress_gainsdevs)
    tex += r'''\end{tabular}
    \end{document}
    '''
    print(tex)

df = pd.read_csv('models/results-nimages.csv')
df = df.fillna(0)
df['class'] = [int(s.split('-')[0][-1]) for s in df['model']]
df['iter'] = [int(s.split('-')[1][-1]) for s in df['model']]
df['rep'] = [int(s.split('-')[2][3]) for s in df['model']]

res = df.groupby(['class', 'iter']).agg({'class': 'first', 'iter': 'first', 'images': 'mean', **{f'f{i}': 'mean' for i in range(10)}})
res.columns = [col[0] if col[1] == 'first' else '_'.join(col).strip() for col in res.columns.values]
for i in range(10):
    res[f'f_{i}'] *= 100

print(res.to_latex(index=False))
