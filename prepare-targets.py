import pandas as pd

from meta import train_dates, target_columns

prev = pd.DataFrame(columns=target_columns)

for dt in train_dates:
    print "Processing %s..." % dt

    cur = pd.read_pickle('cache/raw-%s.pickle' % dt)
    cur = cur[target_columns].fillna(0.0).astype(int)

    new = cur.subtract(prev, fill_value=0).clip(lower=0)
    new.to_pickle('cache/targets-%s.pickle' % dt)

    cur = prev

print "Done."
