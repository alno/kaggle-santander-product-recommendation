import pandas as pd
import numpy as np
import xgboost as xgb

from meta import train_dates, test_date
from util import Dataset


params = {
    'objective': 'reg:linear',
    'eta': 0.02,
    'max_depth': 6,
    'silent': 1
}

n_iter = 700
feature_columns = ['cod_prov', 'sexo', 'segmento']

all_dates = train_dates + [test_date]

print "Preparing renta examples..."

examples = []

for dt in all_dates:
    print "  Loading %s..." % dt

    df = pd.read_pickle('cache/basic-%s.pickle' % dt)[['ncodpers', 'renta'] + feature_columns]
    df.dropna(subset=['renta'], inplace=True)
    df.drop_duplicates(subset=['ncodpers'], keep='last', inplace=True)

    examples.append(df)

examples = pd.concat(examples)
examples.drop_duplicates(subset=['ncodpers'], keep='last', inplace=True)
examples.set_index('ncodpers', inplace=True)

X = examples[feature_columns]
y = np.log(examples['renta'])

print "Fitting renta model..."

dtrain = xgb.DMatrix(X.values, label=y.values, feature_names=X.columns)

#xgb.cv(params, dtrain, 10000, 3, verbose_eval=10)

model = xgb.train(params, dtrain, n_iter)

print "Generating renta values..."

for dt in all_dates:
    print "  Processing %s..." % dt

    basic = pd.read_pickle('cache/basic-%s.pickle' % dt)

    df = pd.DataFrame(index=basic.index)
    df['renta'] = basic['renta']
    df['renta_missing'] = basic['renta'].isnull()

    pred = basic.loc[df['renta_missing'], feature_columns]
    dpred = xgb.DMatrix(pred.values, feature_names=pred.columns)

    df.loc[df['renta_missing'], 'renta'] = np.exp(model.predict(dpred))

    Dataset.save_part(dt, 'renta', df.values.astype(np.float32))

Dataset.save_part_features('renta', list(df.columns))

print "Done."
