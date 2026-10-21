#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/data"
BASE=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "$f.gz" ] || curl -fsSLO "$BASE/$f.gz"
done
echo "done"
