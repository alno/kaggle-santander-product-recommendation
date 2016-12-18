import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date, product_columns

from util import Dataset

past_usage = []

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    index = pd.Index(Dataset.load_part(dt, 'idx'))

    df = pd.DataFrame(0, columns=product_columns, index=index, dtype=np.uint8)

    for pi in range(max(di-4, 0), di-1):
        idx = index.intersection(past_usage[pi].index)

        df.loc[idx] += past_usage[pi].loc[idx]

    df = df.clip(lower=0, upper=1)

    Dataset.save_part(dt, 'product-past-usage', sp.csr_matrix(df.values))

    if dt != test_date:
        past_usage.append(pd.DataFrame(Dataset.load_part(dt, 'products').toarray(), columns=product_columns, index=index))

Dataset.save_part_features('product-past-usage', [c + '_used_before' for c in product_columns])

print "Done."
