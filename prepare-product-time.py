import pandas as pd
import numpy as np

from meta import train_dates, test_date, target_columns

from util import Dataset


all_dates = train_dates + [test_date]

product_time = pd.DataFrame(columns=target_columns, index=[])

for dt in all_dates:
    print "Processing %s..." % dt

    cur = pd.read_pickle('cache/basic-%s.pickle' % dt)
    idx = cur.index.intersection(product_time.index)

    df = pd.DataFrame(-100, columns=target_columns, index=cur.index, dtype=np.int16)
    df.loc[idx] = product_time.loc[idx]

    Dataset.save_part(dt, 'product-time', df.values)

    if dt != test_date:
        trg = cur[target_columns].astype(np.int16)

        df[(df < 0) & (trg > 0)] = 0
        df[(df > 0) & (trg < 1)] = 0

        df += 2*trg-1

        df[df < -100] = -100

        product_time = df

Dataset.save_part_features('product-time', [c + '_time' for c in target_columns])

print "Done."
