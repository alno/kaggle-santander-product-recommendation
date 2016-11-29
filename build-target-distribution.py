import numpy as np
import pandas as pd

from meta import train_dates, target_columns, lb_target_means
from util import Dataset

recs = []
for dt in train_dates[1:]:
    print dt
    trg = Dataset.load_part(dt, 'targets')
    trg_cnt = np.array(trg.sum(axis=1)).flatten()

    rec = {}
    for i, c in enumerate(target_columns):
        rec[c] = trg_cnt[(trg[:, i] > 0).toarray().flatten()].mean()

    recs.append(rec)

df = pd.DataFrame.from_records(recs, index=train_dates[1:])


trg = Dataset.load_part(train_dates[-1], 'targets')
trg_cnt = np.array(trg.sum(axis=1)).flatten()

res = []
res_lb = []

for i, c in enumerate(target_columns):
    act = (1.0 / trg_cnt[(trg[:, i] > 0).toarray().flatten()]).sum() / len(trg_cnt)

    print "%s: %.6f / %.6f" % (c, act, lb_target_means[c])

    res.append(act)
    res_lb.append(lb_target_means[c])

print "%.6f / %.6f" % (sum(res), sum(res_lb))

print "Last month target means"
for i, c in enumerate(target_columns):
    print "%s: %.6f / %.6f" % (c, res[i] * df[c].mean(), trg[:, i].mean())

print "LB target means"
for i, c in enumerate(target_columns):
    print "'%s': %.6f," % (c, lb_target_means[c] * df[c].mean())
