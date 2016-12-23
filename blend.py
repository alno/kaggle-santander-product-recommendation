import pandas as pd
import numpy as np

import datetime

from numba import jit

from meta import target_columns, test_date
from util import load_pickle, Dataset

preds = {
    '20161221-0036-xgb2p-0.0269690': 1.0,
    '20161221-0623-xgb3p-0.0269678': 0.9,
    '20161221-1457-xgb4p-0.0269682': 0.7,
    '20161221-1419-nn1p-0.0269826': 0.1,
    '20161221-0244-nn1-0.0269693': 0.1,
    '20161221-1328-xgb2-0.0269597': 0.2,
}

rank = False


def load_and_combine_preds(dt):
    idx = Dataset.load_part(dt, 'idx')

    if dt != test_date:
        idx = idx[Dataset.load_part(dt, 'existing')]

    df = pd.DataFrame(0.0, columns=target_columns, index=idx)

    for p in preds:
        d = np.load('preds/%s-%s.npy' % (p, dt))

        if rank:
            d = d.argsort().argsort()

        pdf = pd.DataFrame(d, columns=load_pickle('preds/%s-columns.pickle' % p), index=idx) * preds[p]

        for col in pdf.columns:
            df[col] -= pdf[col]

    return np.argsort(df.values, axis=1)[:, :7]


@jit
def apk(actual, pred):
    m = actual.sum()

    if m == 0:
        return 0

    res = 0.0
    hits = 0.0

    for i, col in enumerate(pred):
        if i >= 7:
            break

        if actual[col] > 0:
            hits += 1
            res += hits / (i + 1.0)

    return res / min(m, 7)


def mapk(targets, predictions):
    total = 0.0

    for i, pred in enumerate(predictions):
        total += apk(targets[i], pred)

    return total / len(targets)


def generate_submission(predictions):
    return [' '.join(target_columns[i] for i in p) for p in predictions]


## Validating

val_date = '2016-05-28'
val_preds = load_and_combine_preds(val_date)
val_targets = Dataset.load_part(val_date, 'targets')[Dataset.load_part(val_date, 'existing')].toarray()

val_score = mapk(val_targets, val_preds)

print "Score: %.7f" % val_score

prediction_name = '%s-blend-%.7f' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), val_score)

## Generating submission

test_idx = Dataset.load_part(test_date, 'idx')
test_preds = load_and_combine_preds(test_date)

subm = pd.DataFrame({'ncodpers': test_idx, 'added_products': generate_submission(test_preds)})
subm.to_csv('subm/%s.csv.gz' % prediction_name, index=False, compression='gzip')

print "Done."
