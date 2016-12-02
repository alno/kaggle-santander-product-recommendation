import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date

from util import Dataset

offsets = range(1, 4)
past_sums = []

for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    index = pd.read_pickle('cache/basic-%s.pickle' % dt).index

    df = pd.DataFrame(0, columns=['product_sum_%d' % ofs for ofs in offsets], index=index, dtype=np.uint8)

    for ofs in offsets:
        if di - ofs >= 0:
            idx = index.intersection(past_sums[di-ofs].index)

            df.loc[idx, 'product_sum_%d' % ofs] += past_sums[di-ofs].loc[idx]

    Dataset.save_part(dt, 'product-past-sums', df.values)

    if dt != test_date:
        past_sums.append(pd.Series(np.array(Dataset.load_part(dt, 'products').sum(axis=1)).flatten(), index=index))

Dataset.save_part_features('product-past-sums', list(df.columns))

print "Done."
