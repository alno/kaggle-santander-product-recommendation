import pandas as pd
import numpy as np

from meta import train_dates, test_date

from util import Dataset

n_lags = 3

feature_columns = ['tiprel_1mes', 'indrel_1mes']
res_columns = ["%s_chg_%d" % (col, lag) for lag in xrange(1, n_lags+1) for col in feature_columns]

past_features = []

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    cur = pd.DataFrame(Dataset.load_part(dt, 'manual'), columns=Dataset.get_part_features('manual'), index=Dataset.load_part(dt, 'idx'))
    df = pd.DataFrame(-1, columns=res_columns, index=cur.index, dtype=np.uint8)

    for lag in xrange(1, n_lags+1):
        if di - lag >= 0:
            idx = cur.index.intersection(past_features[di-lag].index)

            for col in feature_columns:
                df.loc[idx, "%s_chg_%d" % (col, lag)] = (past_features[di-lag].loc[idx, col] != cur.loc[idx, col])

    Dataset.save_part(dt, 'feature-lags', df.values)

    if dt != test_date:
        past_features.append(cur)

Dataset.save_part_features('feature-lags', res_columns)

print "Done."
