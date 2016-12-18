import pandas as pd
import numpy as np

from meta import train_dates, test_date

from util import Dataset

residence_codes = {
    'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89,
    'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100,
    'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30,
    'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104,
    'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83,
    'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32,
    'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24,
    'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59,
    'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5,
    'QA': 58, 'MZ': 27
}

residence_groups = {
    'es': ['ES'],
    'eu': ['GB', 'FR', 'DE', 'IT', 'RO', 'CH', 'BE', 'PT', 'NL', 'PL', 'SE', 'AT', 'BG', 'IE', 'FI', 'GR', 'DK', 'NO', 'LU', 'AD', 'CZ', 'SK'],
    'ussr': ['UR', 'UA', 'MD', 'BY'],
    'north_america': ['US', 'CA'],
    'south_america': ['AR', 'MX', 'CO', 'BR', 'VE', 'EC', 'BO', 'PY', 'CL', 'PE', 'CU', 'UY', 'DO', 'HN', 'CR', 'GT', 'SV', 'PR', 'PA'],
}

canals = ['KHE', 'KAT', 'KFC', 'KHQ', 'KHM', 'KFA', 'KHN', 'KHK']


for di, dt in enumerate(train_dates + [test_date]):
    print "Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)

    df = pd.DataFrame(index=basic.index)
    df['residence_code'] = basic['pais_residencia'].fillna(-99).map(residence_codes)

    for res in residence_groups:
        df['residence_%s' % res] = basic['pais_residencia'].isin(residence_groups[res])
    df['residence_other'] = ~basic['pais_residencia'].isin(sum(residence_groups.values(), []))

    Dataset.save_part(dt, 'residence', df.values.astype(np.float32))

Dataset.save_part_features('residence', list(df.columns))

print "Done."
