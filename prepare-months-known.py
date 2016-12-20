import pandas as pd
import numpy as np

from meta import train_dates, test_date

from util import Dataset

all_dates = train_dates + [test_date]
past_indexes = []

for di, dt in enumerate(all_dates):
    print "Processing %s..." % dt

    index = Dataset.load_part(dt, 'idx')

    df = pd.DataFrame(index=index)
    df['months_known'] = 0

    for ofs in range(1, 5):
        if di - ofs >= 0:
            df.loc[df.index.isin(past_indexes[di-ofs]), 'months_known'] = ofs

    Dataset.save_part(dt, 'months-known', df.values.astype(np.float32))

    past_indexes.append(index)

Dataset.save_part_features('months-known', list(df.columns))

print "Done."
