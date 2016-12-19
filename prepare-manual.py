import pandas as pd
import numpy as np

from meta import train_dates, test_date

from util import Dataset

raw_columns = [
    'sexo', 'age',
    'ind_nuevo', 'antiguedad', 'indrel_1mes',
    'indresi', 'indext', 'indfall', 'tipodom',
    'ind_actividad_cliente', 'segmento'
]

map_columns = {
    'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},

    'indrel': {1.0: 0, 99.0: 1, -99: 2},
    'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},

    'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}

canals = ['KHE', 'KAT', 'KFC', 'KHQ', 'KHM', 'KFA', 'KHN', 'KHK']


all_dates = train_dates + [test_date]
past_indexes = []

for di, dt in enumerate(all_dates):
    print "Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)

    df = pd.DataFrame(index=basic.index)

    for col in raw_columns:
        df[col] = basic[col]

    for col in map_columns:
        df[col] = basic[col].fillna(-99).map(map_columns[col])

    for canal in canals:
        df['canal_%s' % canal.lower()] = basic['canal_entrada'] == canal
    df['canal_other'] = ~basic['canal_entrada'].isin(canals)

    df['month'] = pd.to_datetime(dt).month
    df['days_since_reg'] = (pd.to_datetime(dt) - basic['fecha_alta']).map(lambda td: td.days)
    df['months_known'] = 0

    for ofs in range(1, 5):
        if di - ofs >= 0:
            df.loc[df.index.isin(past_indexes[di-ofs]), 'months_known'] = ofs

    Dataset.save_part(dt, 'manual', df.values.astype(np.float32))

    past_indexes.append(basic.index)

Dataset.save_part_features('manual', list(df.columns))

print "Done."
