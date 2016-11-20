import pandas as pd
import numpy as np

import datetime
import time

from numba import jit

from meta import train_dates, test_date, target_columns
from util import Dataset

from kaggle_util import Xgb


min_eval_date = '2016-04-28'
min_eval_date = '2016-05-28'


train_data = None
train_targets = None

model = Xgb({'max_depth': 4}, 50)


def load_data(dt):
    basic_df = pd.read_pickle('cache/basic-%s.pickle' % dt)
    prev_products = Dataset.load_part(dt, 'prev-products')

    # Extract columns from basic
    basic = basic_df[['sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel_1mes', 'indresi', 'indext', 'indfall', 'tipodom', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento']].values

    return basic_df.index, np.hstack((basic, prev_products.toarray()))


def add_to_train(data, targets):
    """ Train on encoded rows and their targets """

    target_sum = targets.sum(axis=1)

    # Select only those where something was added
    data = data[target_sum > 0]
    targets = targets[target_sum > 0]

    global train_data, train_targets

    if train_data is None:
        train_data = data
        train_targets = targets
    else:
        train_data = np.vstack((train_data, data))
        train_targets = np.vstack((train_targets, targets))

    print "    Train data shape: %s" % str(train_data.shape)
    print "    Train targets shape: %s" % str(train_targets.shape)


def predict_row(scores, prev_products):
    pred = []
    for c in np.argsort(-scores):
        if prev_products[c] == 0:
            pred.append(c)

            if len(pred) >= 7:
                break

    return pred


def predict_one_label(c, data, targets=None):
    mean = train_targets[:, c].mean()
    shape = (data.shape[0], 1)

    print "Column %s[%d/%d], mean %.7f..." % (target_columns[c], c+1, len(target_columns), mean)

    if mean < 1e-6:  # Label almost never assigned...
        return np.zeros(shape) + mean

    if targets is None:
        return model.fit_predict(train=(train_data, train_targets[:, c]), test=(data,))['ptest'].reshape(shape)
    else:
        return model.fit_predict(train=(train_data, train_targets[:, c]), val=(data, targets[:, c]))['pval'].reshape(shape)


def predict(data, prev_products, targets=None):
    """ Predict """

    scores = np.hstack([predict_one_label(c, data, targets) for c in xrange(len(target_columns))])

    return [predict_row(scores[i], prev_products[i]) for i in xrange(data.shape[0])]


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


def generate_submission(pred):
    return [' '.join(target_columns[i] for i in p) for p in pred]


map_score = None

start_time = time.time()


for dt in train_dates[1:]:  # Skipping first date
    print "%ds, processing %s..." % (time.time() - start_time, dt)
    targets = Dataset.load_part(dt, 'targets').toarray()
    idx, data = load_data(dt)

    if dt >= min_eval_date:
        prev_products = Dataset.load_part(dt, 'prev-products').toarray()

        print "  Predicting..."

        predictions = predict(data, prev_products, targets)
        #print predictions
        map_score = mapk(targets, predictions)

        print "  MAP@7: %.7f" % map_score

    print "  Adding to train..."
    add_to_train(data, targets)


pred_name = 'ml-%s-%.7f' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), map_score)


if True:
    print "Processing test..."

    idx, data = load_data(test_date)

    prev_products = Dataset.load_part(test_date, 'prev-products').toarray()

    print "  Predicting..."
    pred = predict(data, prev_products)

    subm = pd.DataFrame({'ncodpers': idx, 'added_products': generate_submission(pred)})
    subm.to_csv('subm/%s.csv.gz' % pred_name, index=False, compression='gzip')

print "Prediction name: %s" % pred_name
print "Done."
