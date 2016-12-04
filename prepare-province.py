import pandas as pd

from meta import train_dates, test_date
from util import Dataset

provincies = ['MADRID', 'BARCELONA', 'VALENCIA', 'SEVILLA']
all_dates = train_dates + [test_date]

print "Preparing province stats..."

stats = []

for dt in all_dates:
    print "  Loading %s..." % dt

    stats.append(pd.read_pickle('cache/basic-%s.pickle' % dt)['cod_prov'].fillna(-99).astype(int))

stats = pd.concat(stats).value_counts()

print "Generating province values..."

for dt in all_dates:
    print "  Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)

    df = pd.DataFrame(index=basic.index)
    df['province_code'] = basic['cod_prov'].fillna(-99).astype(int).map(stats)
    df['province_missing'] = basic['cod_prov'].isnull()

    #for prov in provincies:
        #df['province_%s' % prov.lower()] = basic['nomprov'] == prov
    #df['province_other'] = ~basic['nomprov'].isin(provincies)

    Dataset.save_part(dt, 'province', df.values)

Dataset.save_part_features('province', list(df.columns))

print "Done."
