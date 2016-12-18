import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, product_columns

from util import Dataset

add_rm_product_columns = product_columns

add_columns = ['%s_add' % c for c in add_rm_product_columns]
rm_columns = ['%s_rm' % c for c in add_rm_product_columns]

prev = pd.DataFrame(columns=product_columns, index=[], dtype=np.uint8)

added = []
removed = []

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    idx = Dataset.load_part(dt, 'idx')

    add = pd.DataFrame(0, columns=add_columns, index=idx, dtype=np.uint8)
    rm = pd.DataFrame(0, columns=rm_columns, index=idx, dtype=np.uint8)

    for i in range(1, 5):
        if di - i > 0:
            add_idx = add.index.intersection(added[di-i].index)
            rm_idx = rm.index.intersection(removed[di-i].index)

            for col in add_rm_product_columns:
                add.loc[add_idx, '%s_add' % col] += added[di-i].loc[add_idx, col]
                rm.loc[rm_idx, '%s_rm' % col] += removed[di-i].loc[rm_idx, col]

    for i in range(di-5):
        added[i] = None
        removed[i] = None

    Dataset.save_part(dt, 'product-add-times', sp.csr_matrix(add.values.astype(np.uint8)))
    Dataset.save_part(dt, 'product-rm-times', sp.csr_matrix(rm.values.astype(np.uint8)))

    if dt != test_date:
        cur = pd.DataFrame(Dataset.load_part(dt, 'products').toarray(), index=idx, columns=product_columns)
        diff = cur.subtract(prev, fill_value=0)

        added.append(diff.clip(lower=0).astype(np.uint8))
        removed.append((-diff).clip(lower=0).astype(np.uint8))

        prev = cur

Dataset.save_part_features('product-add-times', add_columns)
Dataset.save_part_features('product-rm-times', rm_columns)

print "Done."
