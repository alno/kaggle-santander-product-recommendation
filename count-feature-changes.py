import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date

from util import Dataset

n_lags = 3

feature_columns = ['segmento', 'cod_prov', 'tiprel_1mes', 'indrel_1mes', 'ind_empleado', 'indfall', 'age', 'indrel']
res_columns = ["%s_chg_%d" % (col, lag) for col in feature_columns for lag in xrange(1, n_lags+1)]

past_features = []

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)[feature_columns]

    df = pd.DataFrame(0, columns=res_columns, index=basic.index, dtype=np.uint8)

    for lag in xrange(1, n_lags+1):
        if di - lag >= 0:
            idx = basic.index.intersection(past_features[di-lag].index)

            for col in feature_columns:
                df.loc[idx, "%s_chg_%d" % (col, lag)] = (past_features[di-lag].loc[idx, col] != basic.loc[idx, col])

    #Dataset.save_part(dt, 'product-lags', sp.csr_matrix(df.values))

    print pd.DataFrame({'mean': df.mean(), 'sum': df.sum()})

    if dt != test_date:
        past_features.append(basic)

#Dataset.save_part_features('product-lags', res_columns)

print "Done."
