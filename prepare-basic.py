import pandas as pd
import numpy as np

from meta import raw_data_dtypes


def build_ncodpers_map(df, col):
    return df[['ncodpers', col]].dropna().groupby('ncodpers')[col].first()


def fillna_by_ncodpers(df, col):
    df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), 'ncodpers'].map(build_ncodpers_map(df, col))


date_spec = pd.to_datetime('2015-07-28')


for ds in ['train', 'test']:
    print "Loading %s..." % ds

    df = pd.read_csv('../input/%s_ver2.csv.zip' % ds, dtype=raw_data_dtypes, parse_dates=['fecha_dato', 'fecha_alta', 'ult_fec_cli_1t'])

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

    df['renta'] = df['renta'].replace('NA', np.nan).astype(np.float64)
    df['age'] = df['age'].replace('NA', np.nan).astype(np.float64)

    df['indrel_1mes'] = df['indrel_1mes'].replace('P', 0).fillna(-1).astype(float).astype(np.int8)

    df.loc[df['fecha_dato'] < date_spec, 'antiguedad'] = (df.loc[df['fecha_dato'] < date_spec, 'fecha_dato'] - df.loc[df['fecha_dato'] < date_spec, 'fecha_alta']).map(lambda d: d.days / 30, na_action='ignore')
    df['antiguedad'] = df['antiguedad'].replace('NA', np.nan).fillna(300).astype(np.int32)  # TODO Use smarter NA fill

    df['sexo'] = df['sexo'].replace({'V': 1, 'H': -1})
    df['segmento'] = df['segmento'].map(lambda s: int(s[:2]), na_action='ignore')

    if ds == 'train':
        fillna_by_ncodpers(df, 'sexo')
        fillna_by_ncodpers(df, 'segmento')
        fillna_by_ncodpers(df, 'ind_actividad_cliente')
        fillna_by_ncodpers(df, 'ind_nuevo')

    df['sexo'] = df['sexo'].fillna(0).astype(np.int8)  # TODO Try to estimate customer sex by other params ?
    df['segmento'] = df['segmento'].fillna(2.5).astype(np.float16)
    df['ind_actividad_cliente'] = df['ind_actividad_cliente'].fillna(-1).astype(np.int8)  # TODO Use smarter NA fill
    df['ind_nuevo'] = df['ind_nuevo'].fillna(-1).astype(np.int8)  # TODO Use smarter NA fill

    if ds == 'train':
        df['ind_nomina_ult1'] = df['ind_nomina_ult1'].fillna(0).astype(np.uint8)
        df['ind_nom_pens_ult1'] = df['ind_nom_pens_ult1'].fillna(0).astype(np.uint8)

    print "Saving %s..." % ds
    for dt, group in df.groupby('fecha_dato'):
        group.set_index('ncodpers', inplace=True, drop=False, verify_integrity=True)
        group.to_pickle('cache/basic-%s.pickle' % str(dt.date()))

print "Done."
