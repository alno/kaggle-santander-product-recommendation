import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, target_columns

from util import Dataset

prev = pd.DataFrame(columns=target_columns, index=[])

for dt in train_dates:
    print "Processing %s..." % dt

    cur = pd.read_pickle('cache/basic-%s.pickle' % dt)
    cur = cur[target_columns].fillna(0.0).astype(int)

    new = cur.subtract(prev, fill_value=0).clip(lower=0).astype(np.uint8).loc[cur.index]

    Dataset.save_part(dt, 'targets', sp.csr_matrix(new.values))

    if (cur.index != new.index).sum() > 0:
        raise ValueError

    prev = cur

Dataset.save_part_features('targets', target_columns)

print "Done."
