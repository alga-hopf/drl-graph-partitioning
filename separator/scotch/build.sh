#!/bin/bash

export SCOTCH_DIR=$HOME/local/scotch_6.1.0

rm -rf build
mkdir build
cd build
cmake ../
make VERBOSE=1
