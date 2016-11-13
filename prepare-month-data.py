import pandas as pd

from meta import raw_data_dtypes

for ds in ['train', 'test']:
    print "Loading %s..." % ds

    df = pd.read_csv('../input/%s_ver2.csv' % ds, dtype=raw_data_dtypes, parse_dates=['fecha_dato', 'fecha_alta'])

    print "Saving %s..." % ds
    for dt, group in df.groupby('fecha_dato'):
        group.set_index('ncodpers', inplace=True, drop=False, verify_integrity=True)
        group.to_pickle('cache/raw-%s.pickle' % str(dt.date()))

print "Done."
