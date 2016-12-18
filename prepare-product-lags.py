import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, product_columns

from util import Dataset

product_lag_columns = [
    'ind_ahor_fin_ult1',
    'ind_aval_fin_ult1',
    'ind_cco_fin_ult1',
    'ind_cder_fin_ult1',
    'ind_cno_fin_ult1',
    'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1',
    'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1',
    #'ind_deme_fin_ult1',
    'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1',
    'ind_fond_fin_ult1',
    #'ind_hip_fin_ult1',
    'ind_plan_fin_ult1',
    'ind_pres_fin_ult1',
    'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1',
    'ind_valo_fin_ult1',
    #'ind_viv_fin_ult1',
    'ind_nomina_ult1',
    'ind_nom_pens_ult1',
    'ind_recibo_ult1'
]

n_lags = 5

past_usage = []

res_columns = ["%s_lag_%d" % (col, lag) for lag in xrange(1, n_lags+1) for col in product_lag_columns]

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    index = pd.Index(Dataset.load_part(dt, 'idx'))

    df = pd.DataFrame(0, columns=res_columns, index=index, dtype=np.uint8)

    for lag in xrange(1, n_lags+1):
        if di - lag >= 0:
            idx = index.intersection(past_usage[di-lag].index)

            for col in product_lag_columns:
                df.loc[idx, "%s_lag_%d" % (col, lag)] = past_usage[di-lag].loc[idx, col]

    Dataset.save_part(dt, 'product-lags', sp.csr_matrix(df.values))

    if dt != test_date:
        past_usage.append(pd.DataFrame(Dataset.load_part(dt, 'products').toarray(), columns=product_columns, index=index))

Dataset.save_part_features('product-lags', res_columns)

print "Done."
