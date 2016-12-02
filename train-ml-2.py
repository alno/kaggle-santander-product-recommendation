import pandas as pd
import numpy as np

import datetime
import argparse

from numba import jit

from meta import target_columns, lb_target_means, test_date
from util import Dataset
from sklearn.utils import resample

from kaggle_util import Xgb


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--optimize', action='store_true', help='optimize model params')
parser.add_argument('--threads', type=int, default=4, help='specify thread count')

args = parser.parse_args()

model = Xgb({
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': len(target_columns),
    'nthread': args.threads,
    'max_depth': 6,
    'eta': 0.05,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
}, 200)

param_grid = {'max_depth': (3, 8), 'min_child_weight': (1, 10), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0)}

feature_parts = ['prev-products', 'manual', 'product-past-usage']
feature_names = sum(map(Dataset.get_part_features, feature_parts), [])


train_pairs = [
    (['2015-05-28'], '2016-05-28'),
    (['2015-06-28'], '2016-06-28'),
]

n_bags = 4


def densify(d):
    if hasattr(d, 'toarray'):
        return d.toarray()
    else:
        return d


def load_data(dt):
    data = np.hstack([densify(Dataset.load_part(dt, p)) for p in feature_parts])
    prev_products = Dataset.load_part(dt, 'prev-products').toarray()

    if dt == test_date:
        return data, prev_products, pd.read_pickle('cache/basic-%s.pickle' % dt).index

    exs = Dataset.load_part(dt, 'existing')

    data = data[exs]
    prev_products = prev_products[exs]
    targets = Dataset.load_part(dt, 'targets')[exs].toarray()

    return data, prev_products, targets


def load_train_data(dtt):
    data = []
    targets = []

    for dt in dtt:
        dt_data, _, dt_targets = load_data(dt)

        data.append(dt_data)
        targets.append(dt_targets)

    return np.vstack(data), np.vstack(targets)


def prepare_data(data, targets, target_means=None, random_state=11):
    rs = np.random.RandomState(random_state)

    res_data = []
    res_targets = []

    for c in xrange(len(target_columns)):
        idx = targets[:, c] > 0
        cnt = idx.sum()

        if cnt > 0:
            res_data.append(data[idx])
            res_targets.append(np.full(cnt, c, dtype=np.uint8))
        else:
            res_data.append(np.zeros((0, data.shape[1])))
            res_targets.append(np.zeros(0))

    # Resample data to mimic target distribution
    if target_means is not None:
        target_means = target_means / target_means.sum()
        total = sum(t.shape[0] for t in res_targets)

        for i in xrange(len(target_columns)):
            dt, trg = res_data[i], res_targets[i]
            n_samples = int(total * target_means[i] * 0.8)

            if n_samples > trg.shape[0]:
                dtn, trgn = resample(dt, trg, n_samples=n_samples-trg.shape[0], replace=(n_samples > trg.shape[0] * 2), random_state=rs)

                dt = np.vstack((dt, dtn))
                trg = np.hstack((trg, trgn))
            else:
                dt, trg = resample(dt, trg, n_samples=n_samples, replace=False, random_state=rs)

            res_data[i] = dt
            res_targets[i] = trg

        print ("target", target_means)

    dist = np.array([float(t.shape[0]) for t in res_targets])
    dist /= dist.sum()

    print ("dist", dist)

    return np.vstack(res_data), np.hstack(res_targets)


def predict(train_data, train_targets, data, prev_products, target_means, targets=None):
    """ Predict """

    preds = np.zeros((n_bags, data.shape[0], len(target_columns)))

    for bag in xrange(n_bags):
        print "Training model %d/%d..." % (bag+1, n_bags)
        rs = 17 + 11 * bag

        if targets is None:
            preds[bag] = model.fit_predict(train=prepare_data(train_data, train_targets, target_means, random_state=rs), test=(data,), feature_names=feature_names)['ptest']
        else:
            if args.optimize:
                model.optimize(train=prepare_data(train_data, train_targets, target_means, random_state=rs), val=prepare_data(data, targets, random_state=rs+13), param_grid=param_grid, feature_names=feature_names)

            preds[bag] = model.fit_predict(train=prepare_data(train_data, train_targets, target_means, random_state=rs), val=prepare_data(data, targets, random_state=rs+13), test=(data,), feature_names=feature_names)['ptest']

    # Reshape scores, exclude previously bought products
    scores = np.mean(preds, axis=0) * (1 - prev_products)

    print scores.shape

    for i, c in enumerate(target_columns):
        print "%s: %.6f" % (c, scores[:, i].mean())

    #for i in range(len(target_means)):
    #    scores[:, i] *= target_means[i] / scores[:, i].mean()

    # Convert scores to predictions
    return np.argsort(-scores, axis=1)[:, :8]


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


for dtt, dtp in train_pairs:
    print "Training on %s, predicting on %s..." % (dtt, dtp)
    print "  Loading..."

    train_data, train_targets = load_train_data(dtt)

    if dtp != test_date:
        eval_data, eval_prev_products, eval_targets = load_data(dtp)

        print "  Predicting..."

        eval_predictions = predict(train_data, train_targets, eval_data, eval_prev_products, eval_targets.mean(axis=0), eval_targets)

        map_score = mapk(eval_targets, eval_predictions)

        print "  MAP@7: %.7f" % map_score

        del eval_data, eval_targets, eval_prev_products, eval_predictions
    else:
        test_data, test_prev_products, test_idx = load_data(dtp)
        test_target_means = np.array([lb_target_means[c] for c in target_columns])

        print "  Predicting..."

        test_predictions = predict(train_data, train_targets, test_data, test_prev_products, test_target_means)
        test_prediction_name = 'ml-%s-%.7f' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), map_score)

        subm = pd.DataFrame({'ncodpers': test_idx, 'added_products': generate_submission(test_predictions)})
        subm.to_csv('subm/%s.csv.gz' % test_prediction_name, index=False, compression='gzip')

    del train_data, train_targets

print "Prediction name: %s" % test_prediction_name
print "Done."
