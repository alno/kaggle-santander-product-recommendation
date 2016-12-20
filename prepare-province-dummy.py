import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date
from util import Dataset

provincies = ['MADRID', 'BARCELONA', 'VALENCIA', 'SEVILLA']
all_dates = train_dates + [test_date]

print "Preparing province stats..."

values = []

for dt in all_dates:
    print "  Loading %s..." % dt

    values.append(pd.read_pickle('cache/basic-%s.pickle' % dt)['cod_prov'].fillna(-99).astype(int).unique())

values = np.unique(np.array(values).flatten())

print "Generating province values..."

for dt in all_dates:
    print "  Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)
    province = basic['cod_prov'].fillna(-99).astype(int)

    res = np.zeros((basic.shape[0], len(values)), dtype=np.uint8)

    for i, v in enumerate(values):
        res[:, i] = (province == v).astype(np.uint8)

    Dataset.save_part(dt, 'province-dummy', sp.csr_matrix(res))

Dataset.save_part_features('province-dummy', ["province_%s" % v for v in values])

print "Done."
