import logging
import os
import time

import numpy as np
from rtree import index

from src.experiment.common_utils import load_data, Distribution, load_query
from src.spatial_index import SpatialIndex

PAGE_SIZE = 4096


class RTree(SpatialIndex):
    """
    R-tree
    Implement from pypi: rtree
    """

    def __init__(self, model_path=None):
        super(RTree, self).__init__("RTree")
        self.model_path = model_path
        self.index = None
        self.fill_factor = None
        self.leaf_node_capacity = None
        self.non_leaf_node_capacity = None
        self.buffering_capacity = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # for compute
        self.io_cost = 0

    def insert_single(self, point):
        self.index.insert(point[-1], (point[0], point[1]))

    def delete(self, point):
        self.index.delete(point.key, (point.lng, point.lat))

    def build(self, data_list, fill_factor, leaf_node_capacity, non_leaf_node_capacity, buffering_capacity):
        self.fill_factor = fill_factor
        self.leaf_node_capacity = leaf_node_capacity
        self.non_leaf_node_capacity = non_leaf_node_capacity
        self.buffering_capacity = buffering_capacity
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        if buffering_capacity:
            p.buffering_capacity = buffering_capacity
        p.pagesize = PAGE_SIZE
        p.fill_factor = fill_factor
        p.leaf_capacity = leaf_node_capacity
        p.index_capacity = non_leaf_node_capacity
        self.index = index.Index(os.path.join(self.model_path, 'rtree'), properties=p, overwrite=True)
        # self.index = index.RtreeContainer(properties=p)
        self.insert(data_list)

    def point_query_single(self, point):
        """
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return list(self.index.intersection((point[0], point[1])))

    def range_query_single(self, window):
        return list(self.index.intersection((window[2], window[0], window[3], window[1])))

    def knn_query_single(self, knn):
        return list(self.index.nearest((knn[0], knn[1]), int(knn[2])))[:int(knn[2])]

    def save(self):
        rtree_meta = (self.fill_factor, self.leaf_node_capacity, self.non_leaf_node_capacity, self.buffering_capacity)
        np.save(os.path.join(self.model_path, 'rtree_meta.npy'),
                np.array(rtree_meta, dtype=[("0", 'f8'), ("1", 'i1'), ("2", 'i1'), ("3", 'i2')]))

    def load(self):
        rtree_meta = np.load(os.path.join(self.model_path, 'rtree_meta.npy'), allow_pickle=True).item()
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        p.pagesize = 4096
        p.fill_factor = rtree_meta[0]
        p.leaf_capacity = rtree_meta[1]
        p.index_capacity = rtree_meta[2]
        if rtree_meta[3]:
            p.buffering_capacity = rtree_meta[3]
        self.fill_factor = rtree_meta[0]
        self.leaf_node_capacity = rtree_meta[1]
        self.non_leaf_node_capacity = rtree_meta[2]
        self.buffering_capacity = rtree_meta[3]
        self.index = index.Index(os.path.join(self.model_path, 'rtree'), properties=p, overwrite=False)

    def size(self):
        """
        ie_size = data_len * data_size
        structure_size = rtree.data + rtree_meta.npy - ie_size
        """
        size = os.path.getsize(os.path.join(self.model_path, "rtree.data")) + \
               os.path.getsize(os.path.join(self.model_path, "rtree_meta.npy")) - 128
        data_len = len(self.index)
        data_size = 20
        ie_size = data_len * data_size
        structure_size = size - ie_size
        return structure_size, ie_size


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/rtree_skew/"
    data_distribution = Distribution.SKEW
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = RTree(model_path=model_path)
    index_name = index.name
    load_index_from_file = False
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        # pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y)=8, 1 page store 1 node
        # leaf node store xyxy data、pointer、Pointer to next leaf node
        # leaf_node_capacity=(pagesize-size(pointer))/(size(x)*4+size(pointer))=(4096-4)/(8*4+4*1)=113
        # The non leaf node holds the MBR, a pointer to the corresponding child node of the MBR
        # non_leaf_node_capacity = pagesize/(size(x)*4+size(pointer))=4096/(8*4+4*1)=113
        # Due to the presence of fill_factor, the amount of non-leaf node data is between [node_capacity*fill_factor, node_capacity],
        # and the amount of root and leaf node data is unbounded
        # 10w，[0.7, 113, 113]：
        # The average data of non-leaf nodes is about 0.85*113=96, the number of high three layers is 1-96-leaf,
        # the maximum number of leaf nodes is 113*113=12,769, the minimum is 1*79=79
        # The data are uniformly distributed then 10w/113个=885，
        # The data are uniformly distributed then 10w/113*2=1770
        # Single scan IO = tree height = 3
        # index size = (1+96+leafnode)*4096=(1+96+1770)*4096
        # 1451w，[0.7, 113, 113]：
        # Tree height 1-96-96*96-leaf，Assuming extreme aggregation，Then the leaf nodes are 1451w/113=128408
        # The data are uniformly distributed then 10w/113*2=256815
        # Single scan IO = tree height = 4
        # index size =(1+96+96*96+256815)*4096
        index.build(data_list=build_data_list,
                    fill_factor=0.7,
                    leaf_node_capacity=113,
                    non_leaf_node_capacity=113,
                    buffering_capacity=0)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    io_cost = 0
    path = '../../data/query/point_query_skew.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    range_query_list = load_query(data_distribution, 1).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(range_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    knn_query_list = load_query(data_distribution, 2).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(knn_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.SKEW, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))
    logging.info("Update io cost: %s" % (index.io_cost - io_cost))
    io_cost = index.io_cost
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result1.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
