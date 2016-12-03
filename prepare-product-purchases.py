import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, target_columns

from util import Dataset

purchases = pd.DataFrame(columns=target_columns, index=[])

for dt in train_dates + [test_date]:
    print "Processing %s..." % dt

    index = pd.Index(Dataset.load_part(dt, 'idx'))

    idx = index.intersection(purchases.index)

    df = pd.DataFrame(0, columns=target_columns, index=index, dtype=np.uint8)
    df.loc[idx] = purchases.loc[idx]

    Dataset.save_part(dt, 'product-purchases', sp.csr_matrix(df.values))

    if dt != test_date:
        purchases = df
        purchases += pd.DataFrame(Dataset.load_part(dt, 'targets').toarray(), columns=target_columns, index=index)

Dataset.save_part_features('product-purchases', [c + '_pur_cnt' for c in target_columns])

print "Done."
