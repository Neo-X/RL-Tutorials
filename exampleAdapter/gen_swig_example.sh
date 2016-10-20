#!/bin/bash
swig3.0 -Wall -v -debug-classes -c++ -python -o $1example_wrap.cpp $1example.swig