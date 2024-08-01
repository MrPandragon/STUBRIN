## Introduction

This code repository proposes two indexing methods, STUM and STUBRIN, which aim to improve the query performance and update performance of spatially learnt indexes, and builds on our previous work on SLIBS and SLBRIN. 

1. Basic Structure for Spatial Learned Index，SLIBS: A spatial learned index using first dimensionality reduction and then construction of a unidimensional learned index can be seen in detail in Wang Lijun's PhD thesis
2. Spatial Learned Block Range Index，SLBRIN: Optimise SLIBS using space partitioning method and block range index structure to improve query performance in external memory space,《SLBRIN: A Spatial Learned Index Based on BRIN》](https://www.mdpi.com/2220-9964/12/4/171).
3. Tempo-Spatial updatable Spatial Learned Index，STUSLI: Optimisation of SLIBS using spatio-temporal sequence prediction algorithm to improve the query performance and update performance of the update process.
4. Updatable Spatial Learned Block Range Index，STUBRIN: Optimisation of SLBRIN using TSUM to complement its strengths and focus on inefficient updating due to inefficient model construction.

## Content

* data：
  * table：The original dataset, containing two synthetic datasets, uniform and normal, and the nyct real dataset. The number 1 indicates the condition used to construct, the number 2 indicates the update condition, and 10w indicates the test dataset.
  * index：After processing, the original dataset is downscaled and sorted by Geohash.
  * query：Query conditions, containing point search conditions, range search conditions and k-nearest neighbour search conditions for the three datasets.
  * create_data.py：Data processing code, including data cleansing of nyct datasets, generation of synthetic datasets, etc.
* result
  * create_result*.py：Result processing code, according to the style of different journals out of the figure。
* src
  * experiment
  * proposed_sli：our propose index method
  * si（Spatial Index）：Spatial indexes to be compared
  * sli（Spatial Learned Index）：Spatial learned indexes to be compared
  * utils：Tools like Geohash
  * mlp*.py：Replicating RMI（Recursive Model Indexes）
  * b_tree.py：Replicating the B-tree
  * spatial_index.py：Father class of index
  * ts_predict.py：Predictive Models Implemented for Spatio-Temporal Sequence Forecasting
* requirement.txt

## Usage

1. install requirements

```shell
pip install -r ./requirements.txt
```

2. Running test cases for index

```shell
python ./src/si/r_tree.py
python ./src/proposed_sli/slbrin.py
```

Note:

1. Only a test dataset with 10w of data is available in ./data, the full dataset can be generated under python [create_data.py](https://github.com/MrPandragon/STUBRIN/blob/main/data/create_data.py).
2. To switch datasets, change **Distribution.NYCT_10W_SORTED** to **Distribution.NYCT_SORTED** or [other dataset](https://github.com/MrPandragon/STUBRIN/blob/main/src/experiment/common_utils.py).
