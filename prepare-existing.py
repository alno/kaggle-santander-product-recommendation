import pandas as pd

from meta import train_dates, test_date

from util import Dataset

old_index = []

for dt in train_dates + [test_date]:
    print "Processing %s..." % dt

    index = pd.read_pickle('cache/basic-%s.pickle' % dt).index

    Dataset.save_part(dt, 'existing', index.isin(old_index))

    old_index = index

print "Done."
