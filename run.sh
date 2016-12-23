#!/bin/sh

prepare() {
  echo "Preparing $1..."
  python prepare-$1.py
}

train() {
  echo "Training $1..."
  time python -u train-$1.py | tee logs/$1.log
}


# Prepare data
prepare basic
prepare existing
prepare targets
prepare products
prepare product-lags
prepare product-lag-sums
prepare product-add-rm-times
prepare manual
prepare manual-dummy
prepare months-known
prepare renta
prepare province
prepare province-dummy
prepare feature-lags
prepare feature-lag-diffs

# Basic models
#train ml2
