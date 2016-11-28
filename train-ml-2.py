import pandas as pd
import numpy as np

import datetime
import argparse

from numba import jit

from meta import target_columns, lb_target_means
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

feature_parts = ['prev-products', 'manual']
feature_names = sum(map(Dataset.get_part_features, feature_parts), [])


eval_pairs = [('2015-05-28', '2016-05-28')]
test_pair = ('2015-06-28', '2016-06-28')


def densify(d):
    if hasattr(d, 'toarray'):
        return d.toarray()
    else:
        return d


def load_data(dt):
    idx = pd.read_pickle('cache/basic-%s.pickle' % dt).index
    data = [densify(Dataset.load_part(dt, p)) for p in feature_parts]

    return idx, np.hstack(data)


def prepare_data(data, targets, target_means=None):
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
            n_samples = int(total * target_means[i])

            if n_samples > trg.shape[0]:
                dtn, trgn = resample(dt, trg, n_samples=n_samples-trg.shape[0], replace=(n_samples > trg.shape[0] * 2), random_state=11)

                dt = np.vstack((dt, dtn))
                trg = np.hstack((trg, trgn))
            else:
                dt, trg = resample(dt, trg, n_samples=n_samples, replace=False, random_state=11)

            res_data[i] = dt
            res_targets[i] = trg

        print ("target", target_means)

    dist = np.array([float(t.shape[0]) for t in res_targets])
    dist /= dist.sum()

    print ("dist", dist)

    return np.vstack(res_data), np.hstack(res_targets)


def predict(data, prev_products, target_means, targets=None):
    """ Predict """

    shape = (data.shape[0], len(target_columns))

    if targets is None:
        pred = model.fit_predict(train=prepare_data(train_data, train_targets, target_means), test=(data,), feature_names=feature_names)
    else:
        if args.optimize:
            model.optimize(train=prepare_data(train_data, train_targets, target_means), val=prepare_data(data, targets), param_grid=param_grid, feature_names=feature_names)

        pred = model.fit_predict(train=prepare_data(train_data, train_targets, target_means), val=prepare_data(data, targets), test=(data,), feature_names=feature_names)

    # Reshape scores, exclude previously bought products
    scores = pred['ptest'].reshape(shape) * (1 - prev_products)

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


for dtt, dtp in eval_pairs:
    print "Training on %s, evaluating on %s..." % (dtt, dtp)
    print "  Loading..."

    train_targets = Dataset.load_part(dtt, 'targets').toarray()
    train_idx, train_data = load_data(dtt)

    eval_targets = Dataset.load_part(dtp, 'targets').toarray()
    eval_idx, eval_data = load_data(dtp)
    eval_prev_products = Dataset.load_part(dtp, 'prev-products').toarray()

    print "  Predicting..."

    eval_predictions = predict(eval_data, eval_prev_products, eval_targets.mean(axis=0), eval_targets)

    map_score = mapk(eval_targets, eval_predictions)

    print "  MAP@7: %.7f" % map_score

    del train_idx, train_data, train_targets, eval_idx, eval_data, eval_targets, eval_prev_products, eval_predictions


pred_name = 'ml-%s-%.7f' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), map_score)


if True:
    dtt, dtp = test_pair

    print "Training on %s, predicting on %s..." % (dtt, dtp)
    print "  Loading..."

    train_targets = Dataset.load_part(dtt, 'targets').toarray()
    train_idx, train_data = load_data(dtt)

    test_target_means = np.array([lb_target_means[c] for c in target_columns])
    test_idx, test_data = load_data(dtp)
    test_prev_products = Dataset.load_part(dtp, 'prev-products').toarray()

    print "  Predicting..."

    test_predictions = predict(test_data, test_prev_products, test_target_means)

    subm = pd.DataFrame({'ncodpers': test_idx, 'added_products': generate_submission(test_predictions)})
    subm.to_csv('subm/%s.csv.gz' % pred_name, index=False, compression='gzip')

print "Prediction name: %s" % pred_name
print "Done."
