# [intake_io](https://github.com/bhoeckendorf/intake_io)

Image-focused IO for [Python](https://www.python.org) using [intake](https://intake.readthedocs.io/en/latest/) to wrap data loading backends and [xarray](http://xarray.pydata.org/en/stable/) as target.

[![Documentation Status](https://readthedocs.org/projects/intake-io/badge/?version=latest)](https://intake-io.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/312639381.svg)](https://zenodo.org/badge/latestdoi/312639381)

## Why intake?

- Consistent metadata handling, incl. loading of metadata only.
- YAML [catalogs](https://intake.readthedocs.io/en/latest/catalog.html) to maintain data collections from heterogeneous sources, e.g. for machine learning.
- Ability to save [additional arbitrary metadata](https://intake.readthedocs.io/en/latest/catalog.html#metadata) with YAML catalog entries. Supplements metadata of image header (e.g. provide channel names or set pixel spacing) and is saved with the data instead of in analysis scripts. Also useful to add ground truth annotations or specify where to find them.
- Consistent interface to load partial data, if supported by underlying data structure.
- [Dask](https://dask.org)-[friendly.](https://intake.readthedocs.io/en/latest/quickstart.html?highlight=dask#working-with-dask)
- See more in the [intake documentation](https://intake.readthedocs.io/en/latest/).

## Why xarray?

[Xarray](http://xarray.pydata.org/en/stable/why-xarray.html) elegantly keeps related images (e.g. image and segmentation masks) and metadata in a single data structure.

## Data sources

A number of [data sources](https://intake-io.readthedocs.io/en/latest/datasources.html) are supported, with new ones being added at a slow pace.

## Installation

Please see the [documentation](https://intake-io.readthedocs.io/en/latest/installation.html).
