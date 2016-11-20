import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, target_columns

from util import Dataset


all_dates = train_dates + [test_date]

products = pd.DataFrame(columns=target_columns, index=[])

for dt in all_dates:
    print "Processing %s..." % dt

    cur = pd.read_pickle('cache/basic-%s.pickle' % dt)
    idx = cur.index.intersection(products.index)

    df = pd.DataFrame(0, columns=target_columns, index=cur.index, dtype=np.uint8)
    df.loc[idx] = products.loc[idx]

    Dataset.save_part(dt, 'prev-products', sp.csr_matrix(df.values))

    if dt != test_date:
        products = cur[target_columns]

        Dataset.save_part(dt, 'products', sp.csr_matrix(products.values))

Dataset.save_part_features('products', target_columns)
Dataset.save_part_features('prev-products', target_columns)

print "Done."
