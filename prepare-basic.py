import pandas as pd
import numpy as np

from meta import raw_data_dtypes, test_date, product_columns
from util import Dataset


def build_ncodpers_map(df, col):
    return df[['ncodpers', col]].dropna().groupby('ncodpers')[col].first()


def fillna_by_ncodpers(df, col):
    df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), 'ncodpers'].map(build_ncodpers_map(df, col))


def load_clean_data(ds):
    print "Loading %s..." % ds

    df = pd.read_csv('../input/%s_ver2.csv.zip' % ds, dtype=raw_data_dtypes, parse_dates=['fecha_dato', 'fecha_alta', 'ult_fec_cli_1t'])
    df.dropna(subset=['ind_nuevo'], inplace=True)

    print "Preprocessing %s..." % ds

    df.drop(['conyuemp'], axis=1, inplace=True)  # Almost no distinct values

    for col in df.columns:
        if df.dtypes[col] == np.object:
            df[col] = df[col].str.strip()

    df['ind_empleado'].fillna('N', inplace=True)
    df['pais_residencia'].fillna('ES', inplace=True)

    df['indresi'] = (df['indresi'].fillna('S') == 'S').astype(np.uint8)
    df['indext'] = (df['indext'].fillna('N') == 'S').astype(np.uint8)
    df['indfall'] = (df['indfall'].fillna('N') == 'S').astype(np.uint8)
    df['ind_nuevo'] = df['ind_nuevo'].astype(np.uint8)
    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].astype(np.uint8)
    df['antiguedad'] = df['antiguedad'].astype(np.int32)

    df['indrel_1mes'] = df['indrel_1mes'].replace('P', 0).fillna(-1).astype(float).astype(np.int8)

    return df


def save_group(dt, group):
    print "Saving %s..." % dt

    if dt == test_date:
        group.drop(product_columns, axis=1, inplace=True)
    else:
        for col in product_columns:
            group[col] = group[col].fillna(0).astype(np.uint8)

    group.set_index('ncodpers', inplace=True, drop=False, verify_integrity=True)
    group.to_pickle('cache/basic-%s.pickle' % dt)

    Dataset.save_part(dt, 'idx', group['ncodpers'].values)


df = pd.concat(load_clean_data(ds) for ds in ['train', 'test'])

print "Processing merged data..."

df['renta'] = df['renta'].replace('NA', np.nan).astype(np.float64)
df['age'] = df['age'].replace('NA', np.nan).astype(np.float64)

df['sexo'] = df['sexo'].replace({'V': 1, 'H': -1})
df['segmento'] = df['segmento'].map(lambda s: int(s[:2]), na_action='ignore')

fillna_by_ncodpers(df, 'sexo')
fillna_by_ncodpers(df, 'segmento')

df['sexo'] = df['sexo'].fillna(0).astype(np.int8)  # TODO Try to estimate customer sex by other params ?
df['segmento'] = df['segmento'].fillna(2.5).astype(np.float16)

for dt, group in df.groupby('fecha_dato'):
    save_group(str(dt.date()), group.copy())

print "Done."
