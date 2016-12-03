import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, target_columns

from util import Dataset

n_lags = 4

past_usage = []

res_columns = ["%s_lag_%d" % (col, lag) for lag in xrange(1, n_lags+1) for col in target_columns]

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    index = pd.read_pickle('cache/basic-%s.pickle' % dt).index

    df = pd.DataFrame(0, columns=res_columns, index=index, dtype=np.uint8)

    for lag in xrange(1, n_lags+1):
        if di - lag >= 0:
            idx = index.intersection(past_usage[di-lag].index)

            for col in target_columns:
                df.loc[idx, "%s_lag_%d" % (col, lag)] = past_usage[di-lag].loc[idx, col]

    Dataset.save_part(dt, 'product-lags', sp.csr_matrix(df.values))

    if dt != test_date:
        past_usage.append(pd.DataFrame(Dataset.load_part(dt, 'products').toarray(), columns=target_columns, index=index))

Dataset.save_part_features('product-lags', res_columns)

print "Done."
