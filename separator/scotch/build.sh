#!/bin/bash

export SCOTCH_DIR=$HOME/scotch-v6.1.0/src/libscotch/

rm -rf build
mkdir build
cd build
cmake ../
make VERBOSE=1
