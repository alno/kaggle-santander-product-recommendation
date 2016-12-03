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
prepare manual

#prepare product-time
#prepare product-past-sums
#prepare product-past-usage
#prepare product-purchases

# Basic models
train ml2
