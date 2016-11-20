import pandas as pd
import numpy as np

import operator
import datetime
import time

from collections import defaultdict
from numba import jit

from meta import train_dates, test_date, target_columns
from util import Dataset

min_eval_date = '2016-01-28'


counts = defaultdict(lambda: defaultdict(int))
counts_overall = np.zeros(len(target_columns))


def encode_renta(renta):
    if renta < 45542.97:
        return 1
    elif renta < 57629.67:
        return 2
    elif renta < 68211.78:
        return 3
    elif renta < 78852.39:
        return 4
    elif renta < 90461.97:
        return 5
    elif renta < 103855.23:
        return 6
    elif renta < 120063.00:
        return 7
    elif renta < 141347.49:
        return 8
    elif renta < 173418.36:
        return 9
    elif renta < 234687.12:
        return 10
    else:
        return 11


def encode(data, products):
    sexo = data['sexo'].values
    age = data['age'].values
    segmento = data['segmento'].values
    pais_residencia = data['pais_residencia'].fillna('').values
    nomprov = data['nomprov'].fillna('').values
    ncodpers = data['ncodpers'].values
    antiguedad = data['antiguedad'].values
    ind_nuevo = data['ind_nuevo'].fillna(-1).values
    ind_empleado = data['ind_empleado'].fillna('').values
    ind_actividad_cliente = data['ind_actividad_cliente'].fillna(-1).values
    indresi = data['indresi'].fillna(-1).values
    canal_entrada = data['canal_entrada'].fillna('').values
    renta = data['renta'].map(encode_renta, na_action='ignore').fillna(-1).values

    res = []
    for i in xrange(len(data)):
        enc = []
        enc.append((1, pais_residencia[i], sexo[i], age[i], segmento[i], ind_nuevo[i], ind_empleado[i], ind_actividad_cliente[i], indresi[i], renta[i]))
        enc.append((2, pais_residencia[i], sexo[i], age[i], segmento[i], nomprov[i]))
        enc.append((3, pais_residencia[i], sexo[i], age[i], segmento[i], ncodpers[i]))
        enc.append((4, pais_residencia[i], sexo[i], age[i], segmento[i], antiguedad[i]))
        enc.append((5, pais_residencia[i], sexo[i], age[i], segmento[i], ind_nuevo[i]))
        enc.append((6, pais_residencia[i], sexo[i], age[i], segmento[i], ind_actividad_cliente[i]))
        enc.append((7, pais_residencia[i], sexo[i], age[i], segmento[i], canal_entrada[i]))
        enc.append((8, pais_residencia[i], sexo[i], age[i], segmento[i], ind_nuevo[i], canal_entrada[i]))
        enc.append((9, pais_residencia[i], sexo[i], age[i], segmento[i], ind_empleado[i]))
        enc.append((10, pais_residencia[i], sexo[i], age[i], segmento[i], renta[i]))
        enc.append((11, sexo[i], age[i], segmento[i]))

        #for j in xrange(products.shape[1]):
        #    enc.append((100 + j, pais_residencia[i], sexo[i], age[i], products[i, j]))

        res.append(enc)

    return res


#@jit
def partial_fit(enc, targets):
    """ Train on encoded rows and their targets """

    for i, row in enumerate(enc):
        f = targets.indptr[i]
        t = targets.indptr[i+1]

        if t > f:
            indices = targets.indices[f:t]

            for c in indices:
                counts_overall[c] += 1

                for e in row:
                    counts[e][c] += 1


def predict_row(idx, row, bests, bests_overall, prev_products):
    scores = np.zeros(len(target_columns))

    for h in row:
        if h in bests:
            for i, c in enumerate(bests[h]):
                scores[c] += 24 - i + len(h)

    pred = []
    for c in np.argsort(-scores):
        if scores[c] == 0:
            break

        if prev_products[c] == 0:
            pred.append(c)

            if len(pred) >= 7:
                break

    if len(pred) < 7:
        for c in bests_overall:
            if c not in pred and prev_products[c] == 0:
                pred.append(c)

                if len(pred) >= 7:
                    break

    return pred


def predict(data, prev_products, enc):
    """ Predict """

    bests_overall = np.argsort(-counts_overall)
    bests = {}

    for k in counts:
        bests[k] = map(operator.itemgetter(0), sorted(counts[k].items(), key=operator.itemgetter(1), reverse=True))

    return [predict_row(data.index[i], row, bests, bests_overall, prev_products[i]) for i, row in enumerate(enc)]


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


for dt in train_dates:
    print "%ds, processing %s, %d hash buckets..." % (time.time() - start_time, dt, len(counts))
    targets = Dataset.load_part(dt, 'targets')
    data = pd.read_pickle('cache/basic-%s.pickle' % dt)
    prev_products = Dataset.load_part(dt, 'prev-products').toarray()

    print "  Encoding..."
    encoded = encode(data, prev_products)

    if dt >= min_eval_date:
        print "  Predicting..."

        predictions = predict(data, prev_products, encoded)
        map_score = mapk(targets.toarray(), predictions)

        print "  MAP@7: %.7f" % map_score

    print "  Training..."
    partial_fit(encoded, targets)


pred_name = 'hashes-%s-%.7f' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), map_score)


if True:
    print "Processing test..."

    data = pd.read_pickle('cache/basic-%s.pickle' % test_date)
    prev_products = Dataset.load_part(dt, 'prev-products').toarray()

    print "  Encoding..."
    encoded = encode(data, prev_products)

    print "  Predicting..."
    pred = predict(data, prev_products, encoded)

    subm = pd.DataFrame({'ncodpers': data.index, 'added_products': generate_submission(pred)})
    subm.to_csv('subm/%s.csv.gz' % pred_name, index=False, compression='gzip')

print "Prediction name: %s" % pred_name
print "Done."
