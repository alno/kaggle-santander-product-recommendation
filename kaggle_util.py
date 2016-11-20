import xgboost as xgb


class Xgb(object):

    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'silent': 1,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter

    def fit_predict(self, train, val=None, test=None, seed=42, feature_names=None):
        dtrain = xgb.DMatrix(train[0], label=train[1], feature_names=feature_names)

        watchlist = [(dtrain, 'train')]

        if val is not None:
            dval = xgb.DMatrix(val[0], label=val[1], feature_names=feature_names)
            watchlist.append((dval, 'eval'))

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = dtrain.get_label().mean()

        model = xgb.train(params, dtrain, self.n_iter, watchlist, verbose_eval=10)

        print "    Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(model.get_fscore().items(), key=lambda t: -t[1]))

        res = {}

        if val is not None:
            res['pval'] = model.predict(dval)

        if test is not None:
            dtest = xgb.DMatrix(test[0], feature_names=feature_names)
            res['ptest'] = model.predict(dtest)

        return res
