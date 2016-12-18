import xgboost as xgb
import numpy as np

from bayes_opt import BayesianOptimization


class Xgb(object):

    default_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'silent': 1,
    }

    def __init__(self, params, n_iter=400):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter

    def fit_predict(self, train, val=None, test=None, feature_names=None, seed=42):
        print "    Train data shape: %s" % str(train[0].shape)

        dtrain = xgb.DMatrix(train[0], label=train[1], feature_names=feature_names)

        watchlist = [(dtrain, 'train')]

        if val is not None:
            dval = xgb.DMatrix(val[0], label=val[1], feature_names=feature_names)
            watchlist.append((dval, 'eval'))

        params = self.params.copy()
        params['seed'] = seed

        model = xgb.train(params, dtrain, self.n_iter, watchlist, verbose_eval=10)
        model.dump_model('xgb.dump', with_stats=True)

        print "    Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(model.get_fscore().items(), key=lambda t: -t[1]))

        del dtrain

        res = {}

        if val is not None:
            print "    Eval data shape: %s" % str(val[0].shape)
            res['pval'] = model.predict(dval)
            del dval

        if test is not None:
            print "    Test data shape: %s" % str(test[0].shape)
            res['ptest'] = np.zeros((test[0].shape[0], params['num_class']))

            start = 0
            batch_size = 30000

            while start < test[0].shape[0]:
                dtest = xgb.DMatrix(test[0][start:start+batch_size], feature_names=feature_names)
                res['ptest'][start:start+batch_size] = model.predict(dtest)
                start += batch_size

        return res

    def optimize(self, train, val, param_grid, feature_names=None, seed=42):
        dtrain = xgb.DMatrix(train[0], label=train[1], feature_names=feature_names)
        deval = xgb.DMatrix(train[0], label=train[1], feature_names=feature_names)

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print "Trying %s..." % str(params)

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], verbose_eval=10, early_stopping_rounds=30)

            print "Score %.5f at iteration %d" % (model.best_score, model.best_iteration)
            print "Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(model.get_fscore().items(), key=lambda t: -t[1]))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print "Best score: %.5f, params: %s" % (-opt.res['max']['max_val'], opt.res['mas']['max_params'])
