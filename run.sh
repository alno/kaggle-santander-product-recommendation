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
prepare targets
prepare products
prepare product-time
prepare product-purchases

# Basic models
train hashes
