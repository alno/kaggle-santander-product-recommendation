import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras_util import batch_generator

from sklearn.preprocessing import StandardScaler


class Keras(object):

    def __init__(self, arch, params, scale=True, loss='categorical_crossentropy', n_classes=None):
        self.arch = arch
        self.params = params
        self.scale = scale
        self.loss = loss
        self.n_classes = n_classes

    def fit_predict(self, train, val=None, test=None, feature_names=None, seed=42):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)

        if self.scale:
            scaler = StandardScaler(with_mean=False)

            X_train = scaler.fit_transform(train[0])
            y_train = train[1]

            if val is not None:
                X_eval = scaler.transform(val[0])
                y_eval = val[1]

            if test is not None:
                X_test = scaler.transform(test[0])
        else:
            X_train = train[0]
            y_train = train[1]

            if val is not None:
                X_eval = val[0]
                y_eval = val[1]

            if test is not None:
                X_test = test[0]

        model = self.arch((X_train.shape[1],), params, n_classes=self.n_classes)
        model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss)

        callbacks = list(params.get('callbacks', []))

        model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True, n_classes=self.n_classes), samples_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800, n_classes=self.n_classes) if val is not None else None, nb_val_samples=X_eval.shape[0] if val is not None else None,
            nb_epoch=params['n_epoch'], verbose=1, callbacks=callbacks)

        res = {}

        if val is not None:
            print "    Eval data shape: %s" % str(X_eval.shape)
            res['pval'] = model.predict(X_eval)

        if test is not None:
            print "    Test data shape: %s" % str(X_test.shape)
            res['ptest'] = model.predict(X_test)

        return res


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_lr(input_shape, params, n_classes):
    model = Sequential()
    model.add(Dense(n_classes, input_shape=input_shape, activation='softmax'))

    return model


def nn_mlp_2(input_shape, params, n_classes):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(n_classes, activation='softmax'))

    return model
