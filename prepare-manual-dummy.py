import pandas as pd
import numpy as np

import scipy.sparse as sp

from meta import train_dates, test_date

from util import Dataset

map_columns = {
    'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},

    'indrel': {1.0: 0, 99.0: 1, -99: 2},
    'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},
}

canals = ['KHE', 'KAT', 'KFC', 'KFA', 'KHK', 'KHQ', 'KHM', 'KHD', 'KHN', 'KAS', 'RED', 'KAG']


all_dates = train_dates + [test_date]

for di, dt in enumerate(all_dates):
    print "Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)

    df = pd.DataFrame(index=basic.index)
    df['age_less_20'] = basic['age'] < 20

    for col in map_columns:
        values = basic[col].fillna(-99).map(map_columns[col])

        for k in map_columns[col]:
            df['%s_%s' % (col, k)] = values == map_columns[col][k]

    for canal in canals:
        df['canal_%s' % canal.lower()] = basic['canal_entrada'] == canal
    df['canal_other'] = ~basic['canal_entrada'].isin(canals)

    df['month_is_6'] = pd.to_datetime(dt).month == 6

    Dataset.save_part(dt, 'manual-dummy', sp.csr_matrix(df.values.astype(np.uint8)))

Dataset.save_part_features('manual-dummy', list(df.columns))

print "Done."
