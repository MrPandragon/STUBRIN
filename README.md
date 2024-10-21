## Introduction

This code repository proposes two indexing methods, STUM and STUBRIN, which aim to improve the query performance and update performance of spatially learned indexes, and build SLIBS and SLBRIN for contruction of STUM and STUBRIN . 

1. Basic Structure for Spatial Learned Index，SLIBS: A spatial learned index using dimensionality reduction and then construction of a unidimensional learned index
2. Spatial Learned Block Range Index，SLBRIN: Optimise SLIBS using space partitioning method and block range index structure to improve query performance in external memory space
3. Spatio-Temporal Updatable Method，STUM: Optimisation of SLIBS using spatio-temporal sequence prediction algorithm to improve the query performance and update performance of the update process.
4. Spatio-Temporal Updatable Learned Block Range Index，STUBRIN: Optimisation of SLBRIN using STUM to complement its strengths and focus on inefficient updating due to inefficient model construction.

## Getting Started
### Source Code Info
We implement STUM and STUBRIN with python 3.7 on the Ubuntu. You need to install packages in requirements.txt. The codes can be found here.

### Content

* data：
  * table：The test dataset, containing three synthetic datasets (uniform, normal, skew), and the nyct real dataset. The number 1 indicates the condition used to construct, the number 2 indicates the update condition.
  * index：After processing, the original dataset is downscaled and sorted by Geohash.
  * query：Query conditions, containing point search conditions, range search conditions and k-nearest neighbour search conditions for the three datasets.
  * create_data.py：Data processing code, including data cleansing of nyct datasets, generation of synthetic datasets, etc.
* src
  * experiment
  * proposed_sli：our propose index method
  * sli（Spatial Learned Index）：Spatial learned indexes to be compared
  * utils：Tools like Geohash
  * spatial_index.py：Father class of index
  * ts_predict.py：Predictive Models Implemented for Spatio-Temporal Sequence Forecasting
* requirement.txt

### Usage

1. install requirements

```shell
pip install -r ./requirements.txt
```

2. Running test cases for index

```shell
python ./src/proposed_sli/slibs.py
python ./src/proposed_sli/slbrin.py
python ./src/proposed_sli/stum.py
python ./src/proposed_sli/stubrin.py
```

Note:

1. The full dataset can be generated under python [create_data.py](https://github.com/MrPandragon/STUBRIN/blob/main/data/create_data.py).
2. To switch datasets, change **Distribution.NYCT_10W_SORTED** to **Distribution.NYCT_SORTED** or [other dataset](https://github.com/MrPandragon/STUBRIN/blob/main/src/experiment/common_utils.py).
