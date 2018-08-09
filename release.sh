#!/usr/bin/env bash

# Delete build folder in case it exists
rm -r dist

# Build source distribution
python setup.py sdist

# Upload to PyPi
twine upload dist/*

# Clean up
rm -r dist
rm -r src/rnaseq_lib3.egg-info
