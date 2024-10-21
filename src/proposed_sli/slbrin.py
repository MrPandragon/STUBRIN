import gc
import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_precision, data_region, load_query
from src.mlp import MLP
from src.mlp_simple import MLPSimple
from src.spatial_index import SpatialIndex
from src.utils.common_utils import Region, binary_search_less_max, relu, biased_search_almost, \
    biased_search_duplicate, get_mbr_by_points, merge_sorted_list, intersect, binary_search_duplicate
from src.utils.geohash_utils import Geohash

PAGE_SIZE = 4096
HR_SIZE = 8 + 1 + 2 + 4 + 1  # 16
CR_SIZE = 8 * 4 + 2 + 1  # 35
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class SLBRIN(SpatialIndex):
    """
    Spatial Learned Block Range Index，SLBRIN
    1. Basic idea: solving the spatial partitioning problem and the jumping problem on the basis of BRIN-Spatial
    1.1. Proposing Spatial Equidistant Partitioning Method to Optimize Block-Wide Partitioning Results
    1.2. Optimizing Spatial Retrieval Strategies by Combining Spatial Location Code and Learned Index
    """

    def __init__(self, model_path=None):
        super(SLBRIN, self).__init__("SLBRIN")
        self.index_entries = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # A meta page consists of meta
        # version
        # last_hr: Added: pointer to last hr
        # last_cr: Added: pointer to last hr
        # threshold_number: Added: data range for hr, also a threshold for the number of index entries for hr splits
        # threshold_length: Added: geohash length threshold for hr splits
        # threshold_err: Added: error threshold for hr retraining mods
        # threshold_summary: Added: data range for cr and also a threshold for the number of index entries for mbr
        # threshold_merge: Added: cr number threshold for cr consolidation
        # geohash: Added: corresponds to L = geohash.sum_bits, the actual length of the index item geohash encoding
        self.meta = None
        # History range pages consist of multiple hr pages.
        # value: Alteration: Integer geohash of length L
        # length: Added: actual length of geohash
        # number: Added: amount of data for indexed items in range
        # model: Added: learned indices
        # state: Added: 1=inefficient
        # scope: Optimize the computation required
        # value_diff: Required for optimization calculation: next hr value - hr value
        self.history_ranges = None
        # current range pages consists of multiple cr pages.
        # value: Alteration:mbr
        # number: Added: amount of data for indexed items in range
        # state: Added: status, 1=full, 2=outdated
        self.current_ranges = None
        # for train
        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None
        self.retrain_state = 0
        self.sum_up_full_cr_time = 0.0
        self.merge_outdated_cr_time = 0.0
        self.retrain_inefficient_model_time = 0.0
        self.retrain_inefficient_model_num = 0
        # for compute
        self.io_cost = 0

    def build(self, data_list, is_sorted, threshold_number, data_precision, region, threshold_err,
              threshold_summary, threshold_merge,
              is_new, is_simple, weight, core, train_step, batch_num, learning_rate, use_threshold, threshold,
              retrain_time_limit, thread_pool_size):
        """
        build SLBRIN
        1. order data by geohash
        2. build SLBRIN
        2.1. init hr
        2.2. quartile recursively
        2.3. reorganize index entries
        2.4. create slbrin
        3. build learned model
        """
        self.weight = weight
        self.cores = core
        self.train_step = train_step
        self.batch_num = batch_num
        self.learning_rate = learning_rate
        # 1. order data by geohash
        geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        if is_sorted:
            data_list = data_list.tolist()
        else:
            data_list = [(data_list[i][0], data_list[i][1], geohash.encode(data_list[i][0], data_list[i][1]), i)
                         for i in range(len(data_list))]
            data_list.sort(key=lambda x: x[2])
        # 2. build SLBRIN
        # 2.1. init hr
        n = len(data_list)
        range_stack = [(0, 0, n, 0, region)]
        range_list = []
        threshold_length = region.get_max_depth_by_region_and_precision(precision=data_precision) * 2
        # 2.2. quartile recursively
        while len(range_stack):
            cur = range_stack.pop(-1)
            if cur[2] > threshold_number and cur[1] < threshold_length:
                child_regions = cur[4].split()
                l_key = cur[3]
                r_key = cur[3] + cur[2] - 1
                tmp_l_key = l_key
                child_list = [None] * 4
                length = cur[1] + 2
                r_bound = cur[0]
                for i in range(4):
                    value = r_bound
                    r_bound = cur[0] + (i + 1 << geohash.sum_bits - length)
                    tmp_r_key = binary_search_less_max(data_list, 2, r_bound, tmp_l_key, r_key)
                    child_list[i] = (value, length, tmp_r_key - tmp_l_key + 1, tmp_l_key, child_regions[i])
                    tmp_l_key = tmp_r_key + 1
                range_stack.extend(child_list[::-1])  # Put it in init backwards, keep it in order.
            else:
                # Add the hr's that don't need to be split to the result list, adding them in reverse order of
                # [top-left, bottom-right, top-left, top-right] because the stack
                range_list.append(cur)
        # 2.3. reorganize index entries
        self.index_entries = [data_list[r[3]: r[3] + r[2]] for r in range_list]
        # 2.4. create slbrin
        self.meta = Meta(len(range_list) - 1, -1, threshold_number, threshold_length, threshold_err, threshold_summary,
                         threshold_merge, geohash)
        region_offset = pow(10, -data_precision - 1)
        self.history_ranges = [HistoryRange(r[0], r[1], r[2], None, 0, r[4].up_right_less_region(region_offset),
                                            2 << geohash.sum_bits - r[1] - 1) for r in range_list]
        self.current_ranges = []
        self.create_cr()
        # 3. build learned model
        self.build_nn_multiprocess(is_new, is_simple, weight, core, train_step, batch_num, learning_rate,
                                   use_threshold, threshold, retrain_time_limit, thread_pool_size)

    def build_nn_multiprocess(self, is_new, is_simple, weight, core, train_step, batch_num, learning_rate,
                              use_threshold, threshold, retrain_time_limit, thread_pool_size):
        model_hdf_dir = os.path.join(self.model_path, "hdf/")
        if os.path.exists(model_hdf_dir) is False:
            os.makedirs(model_hdf_dir)
        model_png_dir = os.path.join(self.model_path, "png/")
        if os.path.exists(model_png_dir) is False:
            os.makedirs(model_png_dir)
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()
        for hr_key in range(self.meta.last_hr + 1):
            hr = self.history_ranges[hr_key]
            # Training data is bottom left point + partitioned data + top right point
            inputs = [ie[2] for ie in self.index_entries[hr_key]]
            inputs.insert(0, hr.value)
            inputs.append(hr.value + hr.value_diff)
            data_num = hr.number + 2
            labels = list(range(data_num))
            batch_size = 2 ** math.ceil(math.log(data_num / batch_num, 2))
            if batch_size < 1:
                batch_size = 1
            # batch_size = batch_num
            pool.apply_async(build_nn, (self.model_path, hr_key, inputs, labels, is_new, is_simple,
                                        weight, core, train_step, batch_size, learning_rate,
                                        use_threshold, threshold, retrain_time_limit, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.history_ranges[key].model = value

    def insert_single(self, point):
        # 1. encode p to geohash and create index entry(x, y, geohash, t, pointer)
        point = (point[0], point[1], self.meta.geohash.encode(point[0], point[1]), point[2], point[3])
        # 2. insert into cr
        self.current_ranges[-1].number += 1
        self.index_entries[-1].append(tuple(point))
        # 3. parallel transactions
        self.get_sum_up_full_cr()
        self.get_merge_outdated_cr()

    def insert(self, points):
        # Reset counting and timing properties
        self.sum_up_full_cr_time = 0.0
        self.merge_outdated_cr_time = 0.0
        self.retrain_inefficient_model_time = 0.0
        self.retrain_inefficient_model_num = 0
        points = points.tolist()
        for point in points:
            self.insert_single(point)
        # Theoretically retraining occurs at get_retrain_inefficient_model,
        # but the number of retrains rises exponentially once TS and TM become small,
        # so it can be done at the end of each insertion task
        self.post_retrain_inefficient_model()

    def create_cr(self):
        self.current_ranges.append(CurrentRange(value=None, number=0, state=0))
        self.meta.last_cr += 1
        self.index_entries.append([])

    def get_sum_up_full_cr(self):
        """
        Listen for the number of last cr, if it exceeds ts_summary, set it to full and append a new cr.
        """
        if self.current_ranges[-1].number >= self.meta.threshold_summary:
            self.current_ranges[-1].state = 1
            self.create_cr()
            self.post_sum_up_full_cr()

    def post_sum_up_full_cr(self):
        """
        Get all full's cr, stats MBR
        """
        # full_crs = [cr for cr in self.current_ranges if cr.state == 1]
        cr_key = self.meta.last_cr
        while cr_key >= 0:
            cr = self.current_ranges[cr_key]
            if cr.state == 1:
                start_time = time.time()
                cr.value = get_mbr_by_points(self.index_entries[self.meta.last_hr + 1])
                cr.state = 0
                end_time = time.time()
                self.sum_up_full_cr_time += end_time - start_time
                break
            cr_key -= 1

    def get_merge_outdated_cr(self):
        """
        Listen to the number of crs, and if it exceeds ts_merge(outdated), set the first ts_merge crs to outdated.
        """
        if self.meta.last_cr >= self.meta.threshold_merge:
            for cr in self.current_ranges[:self.meta.threshold_merge]:
                cr.state = 2
            # self.post_merge_outdated_cr()

    def post_merge_outdated_cr(self):
        """
        Get all outdated crs, merge the first ts_merge crs into hr, and delete the objects corresponding to those crs.
        """
        # outdated_crs = [cr for cr in self.current_ranges if cr.state == 2]
        if self.current_ranges[0].state == 2:
            start_time = time.time()
            # 1. order index entries in outdated crs(first ts_merge cr)
            old_data_len = self.meta.threshold_merge * self.meta.threshold_summary
            first_cr_key = self.meta.last_hr + 1
            old_data = []
            for range_ies in self.index_entries[first_cr_key:first_cr_key + self.meta.threshold_merge]:
                old_data.extend(range_ies)
            old_data.sort(key=lambda x: x[2])
            # 2. merge index entries into hrs
            hr_num = self.meta.last_hr + 1
            bks = [0] * hr_num
            self.split_data_by_hr(old_data, 0, self.meta.last_hr, 0, old_data_len - 1, bks)
            tmp_bk = 0
            offset = 0  # The presence of split_hr in update_hr causes subsequent hr_keys to be offset backward,
            # so offset is recorded with offset
            for i in range(hr_num):
                bk = bks[i]
                if bk and bk > tmp_bk:  # The practice of taking mid in split causes part of the right boundary
                    # to appear in the previous hr,
                    offset += self.update_hr(i + offset, old_data[tmp_bk:bk])
                    tmp_bk = bk
            # 3. delete crs/index entries
            del self.current_ranges[:self.meta.threshold_merge]
            self.meta.last_cr -= self.meta.threshold_merge
            first_cr_key += offset
            del self.index_entries[first_cr_key:first_cr_key + self.meta.threshold_merge]
            end_time = time.time()
            self.merge_outdated_cr_time += end_time - start_time

    def split_data_by_hr(self, data, l_hr_key, r_hr_key, l_data_key, r_data_key, result):
        m_hr_key = (l_hr_key + r_hr_key) // 2
        m_data_key = binary_search_less_max(data, 2, self.history_ranges[m_hr_key].value, l_data_key, r_data_key)
        result[m_hr_key - 1] = m_data_key + 1
        if m_data_key >= l_data_key:
            if m_hr_key > l_hr_key:
                self.split_data_by_hr(data, l_hr_key, m_hr_key - 1, l_data_key, m_data_key, result)
        if m_data_key < r_data_key:
            if m_hr_key < r_hr_key:
                self.split_data_by_hr(data, m_hr_key + 1, r_hr_key, m_data_key + 1, r_data_key, result)

    def get_retrain_inefficient_model(self, hr_key, old_err):
        """
        Listen for error ranges during model updates, and if they exceed ts_err(inefficient), set them to the inefficient state
        """
        hr = self.history_ranges[hr_key]
        if hr.model.max_err - hr.model.min_err > self.meta.threshold_err * old_err:
            hr.state = 1
            self.retrain_state = 1
            # self.retrain_inefficient_model(hr_key)

    def retrain_inefficient_model(self, hr_key):
        """
        Retraining individual inefficiency state HRs
        """
        start_time = time.time()
        hr = self.history_ranges[hr_key]
        inputs = [j[2] for j in self.index_entries[hr_key]]
        inputs.insert(0, hr.value)
        inputs.append(hr.value + hr.value_diff)
        data_num = hr.number + 2
        labels = list(range(data_num))
        batch_size = 2 ** math.ceil(math.log(data_num / self.batch_num, 2))
        if batch_size < 1:
            batch_size = 1
        tmp_index = NN(self.model_path, str(hr_key), inputs, labels, True, self.weight,
                       self.cores, self.train_step, batch_size, self.learning_rate, False, None, None)
        # tmp_index.build_simple(None)
        tmp_index.build_simple(hr.model.matrices)
        hr.model = AbstractNN(tmp_index.matrices, hr.model.hl_nums,
                              math.floor(tmp_index.min_err),
                              math.ceil(tmp_index.max_err))
        hr.state = 0
        end_time = time.time()
        self.retrain_inefficient_model_time += end_time - start_time
        self.retrain_inefficient_model_num += 1

    def post_retrain_inefficient_model(self):
        """
        Retrain all HRs in inefficient states
        """
        if self.retrain_state:
            for i in range(self.meta.last_hr + 1):
                if self.history_ranges[i].state:
                    self.retrain_inefficient_model(i)
            self.retrain_state = 0

    def update_hr(self, hr_key, points):
        """
        update hr by points
        """
        # merge cr data into hr data
        hr = self.history_ranges[hr_key]
        # quicksort->merge_sorted_array->sorted->binary_search and insert => 50:2:1:0.5
        hr_data = self.index_entries[hr_key]
        merge_sorted_list(hr_data, points)
        hr_number = len(hr_data)
        if hr_number > self.meta.threshold_number and hr.length < self.meta.threshold_length:
            # split hr
            return self.split_hr(hr, hr_key, hr_data) - 1
        else:
            # update hr metadata
            hr.number = hr_number
            hr.max_key = hr_number - 1
            old_err = hr.model.max_err - hr.model.min_err
            hr.update_error_range(hr_data)
            self.get_retrain_inefficient_model(hr_key, old_err)
            return 0

    def split_hr(self, hr, hr_key, hr_data):
        # 1. create child hrs, of which model is inherited from parent hr and update err by inherited index entries
        region_offset = pow(10, -self.meta.geohash.data_precision - 1)
        range_stack = [(hr.value, hr.length, len(hr_data), 0, hr.scope.up_right_more_region(region_offset), hr.model)]
        range_list = []
        while len(range_stack):
            cur = range_stack.pop(-1)
            if cur[2] > self.meta.threshold_number and cur[1] < self.meta.threshold_length:
                child_regions = cur[4].split()
                l_key = cur[3]
                r_key = cur[3] + cur[2] - 1
                tmp_l_key = l_key
                child_list = [None] * 4
                length = cur[1] + 2
                r_bound = cur[0]
                child_matrices_list = cur[5].splits()
                for i in range(4):
                    value = r_bound
                    r_bound = cur[0] + (i + 1 << self.meta.geohash.sum_bits - length)
                    tmp_r_key = binary_search_less_max(hr_data, 2, r_bound, tmp_l_key, r_key)
                    child_list[i] = (value, length, tmp_r_key - tmp_l_key + 1, tmp_l_key, child_regions[i],
                                     child_matrices_list[i])
                    tmp_l_key = tmp_r_key + 1
                range_stack.extend(child_list[::-1])
            else:
                range_list.append(cur)
        child_len = len(range_list)
        child_ranges = []
        for r in range_list:
            child_data = hr_data[r[3]:r[3] + r[2]]
            child_hr = HistoryRange(r[0], r[1], r[2], r[5], 0, r[4].up_right_less_region(region_offset),
                                    2 << self.meta.geohash.sum_bits - r[1] - 1)
            child_hr.update_error_range(child_data)
            child_ranges.append([child_hr, child_data])
        # 2. replace old hr and data
        del self.index_entries[hr_key]
        del self.history_ranges[hr_key]
        child_ranges.reverse()  # Reverse the order to help insert
        for child_range in child_ranges:
            self.history_ranges.insert(hr_key, child_range[0])
            self.index_entries.insert(hr_key, child_range[1])
        # 3. update meta
        self.meta.last_hr += child_len - 1
        # 4. check model inefficient
        old_err = hr.model.max_err - hr.model.min_err
        for i in range(child_len):
            self.get_retrain_inefficient_model(hr_key + i, old_err)
        return child_len

    def point_query_hr(self, point):
        """
        Find the key of the hr according to geohash.
        1. Compute geohash corresponding to hr's geohash_int
        2. Find the maximum value smaller than geohash_int which is the hr where geohash is located.
        """
        return self.binary_search_less_max(point, 0, self.meta.last_hr)

    def range_query_hr(self, point1, point2):
        """
        Find the key of all hr's between geohash1/geohash2 and their position in relation to the window.
        1. Find all the org_geohashes corresponding to the window and the corresponding window's position by geohash_int1/geohash_int2
        2. Filter org_geohash by prefix match to find tgt_geohash
        3. Group and merge positions according to tgt_geohash
        """
        # 1.
        hr_key1 = self.binary_search_less_max(point1, 0, self.meta.last_hr)
        hr_key2 = self.binary_search_less_max(point2, hr_key1, self.meta.last_hr)
        if hr_key1 == hr_key2:
            return {hr_key1: 15}
        else:
            max_length = max(self.history_ranges[hr_key1].length, self.history_ranges[hr_key2].length)
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, max_length)
            # 2.
            # 3.
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {hr_key1: org_geohash_list[0][1],
                                hr_key2: org_geohash_list[-1][1]}
            while True:
                if self.history_ranges[hr_key1].value > org_geohash_list[i][0]:
                    key = hr_key1 - 1
                    pos = org_geohash_list[i][1]
                    if self.history_ranges[key].length > max_length:
                        tgt_geohash_dict[key] = pos
                        tgt_geohash_dict[key + 1] = pos
                        tgt_geohash_dict[key + 2] = pos
                        tgt_geohash_dict[key + 3] = pos
                    else:
                        tgt_geohash_dict[key] = tgt_geohash_dict.get(key, 0) | org_geohash_list[i][1]
                    i += 1
                    if i >= size:
                        break
                else:
                    hr_key1 += 1
                    if hr_key1 > self.meta.last_hr:
                        tgt_geohash_dict[hr_key2] = tgt_geohash_dict[hr_key2] | org_geohash_list[i][1]
                        break
            return tgt_geohash_dict
            # Prefix matching is too slow: time complexity = O(len(number of geohashes corresponding to window)*(j-i))

    def range_query_blk(self, window):
        """
        Find the cr that may intersect window and its spatial relationship (intersect=1/window contains value=2)
        Containment relationships can speed up queries, i.e., containment means that all data within the cr matches the condition
        """
        return [[cr, intersect(window, cr.value)]
                for cr in self.current_ranges]

    def knn_query_hr(self, center_hr_key, point1, point2, point3):
        """
        Find the key of all hr's between geohash1/geohash2 and their position in relation to the window and sort them based on the distance from point3
        1. Find all the org_geohashes corresponding to the window and the corresponding window's position by geohash_int1/geohash_int2
        2. Filter org_geohash by prefix match to find tgt_geohash
        3. Group and merge positions according to tgt_geohash
        4. Calculate the distance between each tgt_geohash and point3 and sort in descending order
        """
        # step 1
        hr_key1 = self.biased_search_less_max(point1, center_hr_key, 0, center_hr_key)
        hr_key2 = self.biased_search_less_max(point2, center_hr_key, center_hr_key, self.meta.last_hr)
        if hr_key1 == hr_key2:
            return [[hr_key1, 15, 0]]
        else:
            max_length = max(self.history_ranges[hr_key1].length, self.history_ranges[hr_key2].length)
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, max_length)
            # step 2
            # step 3
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {hr_key1: org_geohash_list[0][1],
                                hr_key2: org_geohash_list[-1][1]}
            while True:
                if self.history_ranges[hr_key1].value > org_geohash_list[i][0]:
                    key = hr_key1 - 1
                    pos = org_geohash_list[i][1]
                    if self.history_ranges[key].length > max_length:
                        tgt_geohash_dict[key] = pos
                        tgt_geohash_dict[key + 1] = pos
                        tgt_geohash_dict[key + 2] = pos
                        tgt_geohash_dict[key + 3] = pos
                    else:
                        tgt_geohash_dict[key] = tgt_geohash_dict.get(key, 0) | org_geohash_list[i][1]
                    i += 1
                    if i >= size:
                        break
                else:
                    hr_key1 += 1
                    if hr_key1 > self.meta.last_hr:
                        tgt_geohash_dict[hr_key2] = tgt_geohash_dict[hr_key2] | org_geohash_list[i][1]
                        break
            # step 4
            return sorted([[tgt_geohash,
                            tgt_geohash_dict[tgt_geohash],
                            self.history_ranges[tgt_geohash].scope.get_min_distance_pow_by_point_list(point3)]
                           for tgt_geohash in tgt_geohash_dict], key=lambda x: x[2])

    def binary_search_less_max(self, x, left, right):
        """
        Find the maximum value of a bisector smaller than x
        Optimize: loop->bisection->leftmost match:15->1->0.75
        """
        while left <= right:
            mid = (left + right) // 2
            if self.history_ranges[mid].value <= x:
                # leftmost match
                if self.meta.last_hr == mid or self.history_ranges[mid + 1].value > x:
                    return mid
                left = mid + 1
            else:
                right = mid - 1

    def biased_search_less_max(self, x, mid, left, right):
        """
        Find the maximum value smaller than x by bisecting, specifying the initial mid
        Optimize: bisect->biased bisect:3->1
        """
        while left <= right:
            if self.history_ranges[mid].value <= x:
                if self.meta.last_hr == mid or self.history_ranges[mid + 1].value > x:
                    return mid
                left = mid + 1
            else:
                right = mid - 1
            mid = (left + right) // 2

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of points
        2. find hr within geohash by slbrin.point_query
        3. predict by leaf model
        4. biased search in scope [pre - max_err, pre + min_err]
        5. filter cr by mbr
        """
        # 1. compute geohash from x/y of point
        gh = self.meta.geohash.encode(point[0], point[1])
        # 2. find hr within geohash by slbrin.point_query
        hr_key = self.point_query_hr(gh)
        hr = self.history_ranges[hr_key]
        if hr.number == 0:
            result = []
        else:
            # 3. predict by leaf model
            pre = hr.model_predict(gh)
            target_ies = self.index_entries[hr_key]
            # 4. biased search in scope [pre - max_err, pre + min_err]
            l_bound = max(pre - hr.model.max_err, 0)
            r_bound = min(pre - hr.model.min_err, hr.max_key)
            self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
            result = [target_ies[key][4] for key in biased_search_duplicate(target_ies, 2, gh, pre, l_bound, r_bound)]
        # 5. filter cr by mbr
        for cr_key in range(self.meta.last_cr + 1):
            cr = self.current_ranges[cr_key]
            if cr.number and cr.value[0] <= point[1] <= cr.value[1] and cr.value[2] <= point[0] <= cr.value[3]:
                self.io_cost += math.ceil(cr.number / ITEMS_PER_PAGE)
                result.extend(binary_search_duplicate(self.index_entries[cr_key + 1 + self.meta.last_hr],
                                                      2, gh, 0, cr.number - 1))
        return result

    def range_query_single(self, window):
        """
        1. compute geohash from window_left and window_right
        2. get all relative hrs with key and relationship
        3. get min_geohash and max_geohash of every hr for different relation
        4. predict min_key/max_key by nn
        5. filter all the point of scope[min_key/max_key] by range.contain(point)
        6. filter cr by mbr
        Time-consuming operation: range_query_hr/nn predict/exact filter: 15/24/37.6
        """
        # 1. compute geohash of window_left and window_right
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        # 2. get all relative hrs with key and relationship
        hr_list = self.range_query_hr(gh1, gh2)
        result = []
        # 3. get min_geohash and max_geohash of every hr for different relation
        for hr_key in hr_list:
            hr = self.history_ranges[hr_key]
            if hr.number == 0:  # hr is empty
                continue
            position = hr_list[hr_key]
            hr_data = self.index_entries[hr_key]
            if position == 0:  # window contain hr
                result.extend([ie[4] for ie in hr_data])
                self.io_cost += math.ceil(len(hr_data) / ITEMS_PER_PAGE)
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                # if-elif-else->lambda, 30->4
                gh_new1, gh_new2, compare_func = range_position_funcs[position](hr.scope, window, gh1, gh2,
                                                                                self.meta.geohash)
                # 4 predict min_key/max_key by nn
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    left_key = min(biased_search_almost(hr_data, 2, gh_new1, pre1, l_bound1, r_bound1))
                else:
                    l_bound1 = 0
                    left_key = 0
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.max_key)
                    right_key = max(biased_search_almost(hr_data, 2, gh_new2, pre2, l_bound2, r_bound2)) + 1
                else:
                    r_bound2 = hr.number
                    right_key = hr.number
                # 5 filter all the point of scope[min_key/max_key] by range.contain(point)
                # Optimization: region.contain->compare_func does different judgments for points at different locations: 638->474mil
                result.extend([ie[4] for ie in hr_data[left_key:right_key] if compare_func(ie)])
                self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_PAGE)
        # 6. filter cr by mbr
        for cr_key in range(self.meta.last_cr + 1):
            cr = self.current_ranges[cr_key]
            if cr.number:
                pos = intersect(window, cr.value)
                if pos == 0:
                    continue
                elif pos == 2:
                    self.io_cost += math.ceil(cr.number / ITEMS_PER_PAGE)
                    result.extend([ie[-1] for ie in self.index_entries[cr_key + 1 + self.meta.last_hr]])
                else:
                    self.io_cost += math.ceil(cr.number / ITEMS_PER_PAGE)
                    result.extend([ie[-1] for ie in self.index_entries[cr_key + 1 + self.meta.last_hr]
                                   if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
        return result

    def knn_query_single(self, knn):
        """
        1. get the nearest key of query point
        2. get the nn points to create range query window
        3. filter point by distance
        4. filter cr by mbr
        Time-consuming operations: knn_query_hr/nn predict/exact filter: 6.1/30/40.5
        """
        x, y, k = knn
        k = int(k)
        # step 1
        qp_g = self.meta.geohash.encode(x, y)
        qp_hr_key = self.point_query_hr(qp_g)
        qp_hr = self.history_ranges[qp_hr_key]
        qp_hr_data = self.index_entries[qp_hr_key]
        if qp_hr.number == 0:
            qp_ie_key = 0
            tp_ie_list = []
        else:
            pre = qp_hr.model_predict(qp_g)
            l_bound = max(pre - qp_hr.model.max_err, 0)
            r_bound = min(pre - qp_hr.model.min_err, qp_hr.max_key)
            qp_ie_key = biased_search_almost(qp_hr_data, 2, qp_g, pre, l_bound, r_bound)[0]
            tp_ie_list = [qp_hr_data[qp_ie_key]]
        # step 2
        # Three initial result set selection methods:
        # 1. Choose any k
        # 2. Point query to get p. Pick k/2 before and after p
        # 3.  point query to get p, select k before and after p. dst (initial search range) becomes larger,
        # but jump interference becomes smaller
        cur_ie_key = qp_ie_key + 1
        cur_hr_data = qp_hr_data
        cur_hr = qp_hr
        cur_hr_key = qp_hr_key
        i = k
        while i > 0:
            right_ie_len = cur_hr.number - cur_ie_key + 1
            if right_ie_len >= i:
                tp_ie_list.extend(cur_hr_data[cur_ie_key:cur_ie_key + i])
                break
            else:
                tp_ie_list.extend(cur_hr_data[cur_ie_key:])
                if cur_hr_key == self.meta.last_hr:
                    break
                i -= right_ie_len
                cur_hr_key += 1
                cur_hr_data = self.index_entries[cur_hr_key]
                cur_hr = self.history_ranges[cur_hr_key]
                cur_ie_key = 0
        cur_ie_key = qp_ie_key
        cur_hr_key = qp_hr_key
        cur_hr_data = qp_hr_data
        i = k
        while i > 0:
            left_ie_len = cur_ie_key
            if left_ie_len >= i:
                tp_ie_list.extend(cur_hr_data[cur_ie_key - i:cur_ie_key])
                break
            else:
                tp_ie_list.extend(cur_hr_data[cur_ie_key - left_ie_len:cur_ie_key])
                if cur_hr_key == 0:
                    break
                i -= left_ie_len
                cur_hr_key -= 1
                cur_hr_data = self.index_entries[cur_hr_key]
                cur_ie_key = self.history_ranges[cur_hr_key].number
        tp_list = sorted([[(tp_ie[0] - x) ** 2 + (tp_ie[1] - y) ** 2, tp_ie[4]] for tp_ie in tp_ie_list])[:k]
        dst = tp_list[-1][0]
        if dst == 0:
            return [tp[1] for tp in tp_list]
        dst_pow = dst ** 0.5
        window = [y - dst_pow, y + dst_pow, x - dst_pow, x + dst_pow]
        # Handling of out-of-bounds situations
        self.meta.geohash.region.clip_region(window, self.meta.geohash.data_precision)
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        tp_window_hrs = self.knn_query_hr(qp_hr_key, gh1, gh2, knn)
        tp_list = []
        for tp_window_hr in tp_window_hrs:
            if tp_window_hr[2] > dst:
                break
            hr_key = tp_window_hr[0]
            hr = self.history_ranges[hr_key]
            if hr.number == 0:  # hr is empty
                continue
            position = tp_window_hr[1]
            hr_data = self.index_entries[hr_key]
            if position == 0:  # window contain hr
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]] for ie in hr_data]
                self.io_cost += math.ceil(len(hr_data) / ITEMS_PER_PAGE)
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                gh_new1, gh_new2, compare_func = range_position_funcs[tp_window_hr[1]](hr.scope, window, gh1, gh2,
                                                                                       self.meta.geohash)
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    left_key = min(biased_search_almost(hr_data, 2, gh_new1, pre1, l_bound1, r_bound1))
                else:
                    l_bound1 = 0
                    left_key = 0
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.max_key)
                    right_key = max(biased_search_almost(hr_data, 2, gh_new2, pre2, l_bound2, r_bound2)) + 1
                else:
                    r_bound2 = hr.number
                    right_key = hr.number
                # 3. filter point by distance
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                            for ie in hr_data[left_key:right_key] if compare_func(ie)]
                self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_PAGE)
            if len(tmp_list) > 0:
                tp_list.extend(tmp_list)
                tp_list = sorted(tp_list)[:k]
                dst = tp_list[-1][0]
        # 4. filter cr by mbr
        dst_pow = dst ** 0.5
        window = [y - dst_pow, y + dst_pow, x - dst_pow, x + dst_pow]
        for cr_key in range(self.meta.last_cr + 1):
            cr = self.current_ranges[cr_key]
            if cr.number:
                pos = intersect(window, cr.value)
                if pos == 0:
                    continue
                elif pos == 2:
                    self.io_cost += math.ceil(cr.number / ITEMS_PER_PAGE)
                    tp_list.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                                    for ie in self.index_entries[cr_key + 1 + self.meta.last_hr]])
                else:
                    self.io_cost += math.ceil(cr.number / ITEMS_PER_PAGE)
                    tp_list.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                                    for ie in self.index_entries[cr_key + 1 + self.meta.last_hr]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
        tp_list = sorted(tp_list)[:k]
        return [tp[1] for tp in tp_list]

    def save(self):
        assert self.meta.threshold_number < 2 ** (16 - 1), "threshold_number exceed the store size int16"
        slbrin_meta = np.array((self.meta.last_hr, self.meta.last_cr,
                                self.meta.threshold_number, self.meta.threshold_length,
                                self.meta.threshold_err, self.meta.threshold_summary, self.meta.threshold_merge,
                                self.meta.geohash.data_precision,
                                self.meta.geohash.region.bottom, self.meta.geohash.region.up,
                                self.meta.geohash.region.left, self.meta.geohash.region.right,
                                self.weight, self.train_step, self.batch_num, self.learning_rate),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i2'), ("3", 'i2'), ("4", 'i2'), ("5", 'i2'),
                                      ("6", 'i2'), ("7", 'i1'),
                                      ("8", 'f8'), ("9", 'f8'), ("10", 'f8'), ("11", 'f8'),
                                      ("12", 'f4'), ("13", 'i2'), ("14", 'i2'), ("15", 'f4')])
        slbrin_models = np.array([hr.model for hr in self.history_ranges])
        slbrin_hrs = np.array([(hr.value, hr.length, hr.number, hr.state, hr.value_diff,
                                hr.scope.bottom, hr.scope.up, hr.scope.left, hr.scope.right)
                               for hr in self.history_ranges],
                              dtype=[("0", 'i8'), ("1", 'i1'), ("2", 'i2'), ("3", 'i1'), ("4", 'i8'),
                                     ("5", 'f8'), ("6", 'f8'), ("7", 'f8'), ("8", 'f8')])
        slbrin_crs = []
        for cr in self.current_ranges:
            if cr.value is None:
                cr_list = [-1, -1, -1, -1, cr.number, cr.state]
            else:
                cr_list = [cr.value[0], cr.value[1], cr.value[2], cr.value[3], cr.number, cr.state]
            slbrin_crs.append(tuple(cr_list))
        slbrin_crs = np.array(slbrin_crs, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'f8'), ("3", 'f8'),
                                                 ("4", 'i2'), ("5", 'i1')])
        np.save(os.path.join(self.model_path, 'slbrin_meta.npy'), slbrin_meta)
        np.save(os.path.join(self.model_path, 'slbrin_model_cores.npy'), self.cores)
        np.save(os.path.join(self.model_path, 'slbrin_hrs.npy'), slbrin_hrs)
        np.save(os.path.join(self.model_path, 'slbrin_models.npy'), slbrin_models)
        np.save(os.path.join(self.model_path, 'slbrin_crs.npy'), slbrin_crs)
        index_entries = []
        for ies in self.index_entries:
            index_entries.extend(ies)
        index_entries = np.array(index_entries, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')])
        np.save(os.path.join(self.model_path, 'slbrin_data.npy'), index_entries)

    def load(self):
        slbrin_meta = np.load(os.path.join(self.model_path, 'slbrin_meta.npy'), allow_pickle=True).item()
        slbrin_hrs = np.load(os.path.join(self.model_path, 'slbrin_hrs.npy'), allow_pickle=True)
        slbrin_models = np.load(os.path.join(self.model_path, 'slbrin_models.npy'), allow_pickle=True)
        slbrin_crs = np.load(os.path.join(self.model_path, 'slbrin_crs.npy'), allow_pickle=True)
        index_entries = np.load(os.path.join(self.model_path, 'slbrin_data.npy'), allow_pickle=True)
        region = Region(slbrin_meta[8], slbrin_meta[9], slbrin_meta[10], slbrin_meta[11])
        geohash = Geohash.init_by_precision(data_precision=slbrin_meta[7], region=region)
        self.meta = Meta(slbrin_meta[0], slbrin_meta[1], slbrin_meta[2], slbrin_meta[3], slbrin_meta[4], slbrin_meta[5],
                         slbrin_meta[6], geohash)
        self.cores = np.load(os.path.join(self.model_path, 'slbrin_model_cores.npy'), allow_pickle=True).tolist()
        self.weight = slbrin_meta[12]
        self.train_step = slbrin_meta[13]
        self.batch_num = slbrin_meta[14]
        self.learning_rate = slbrin_meta[15]
        # length from int32 to int, otherwise the bitwise operation will exceed the limit and become negative.
        self.history_ranges = [
            HistoryRange(slbrin_hrs[i][0], int(slbrin_hrs[i][1]), slbrin_hrs[i][2], slbrin_models[i], slbrin_hrs[i][3],
                         Region(slbrin_hrs[i][5], slbrin_hrs[i][6], slbrin_hrs[i][7], slbrin_hrs[i][8]),
                         slbrin_hrs[i][4]) for i in range(len(slbrin_hrs))]
        crs = []
        for i in range(len(slbrin_crs)):
            cr = slbrin_crs[i]
            if cr[0] == -1:
                region = None
            else:
                region = cr[:4]
            crs.append(CurrentRange(region, cr[4], cr[5]))
        self.current_ranges = crs
        index_entries = index_entries.tolist()
        # Build the hr part of the ies
        self.index_entries = []
        offset = 0
        for hr in self.history_ranges:
            self.index_entries.append(index_entries[offset:offset + hr.number])
            offset += hr.number
        # Build the cr part of the ies
        for cr in self.current_ranges:
            self.index_entries.append(index_entries[offset:offset + cr.number])
            offset += cr.number

    def size(self):
        """
        structure_size = slbrin_meta.npy + slbrin_hrs.npy + slbrin_models.npy + slbrin_crs.npy
        ie_size = slbrin_data.npy
        """
        # 实际上：
        # meta=os.path.getsize(os.path.join(self.model_path, "slbrin_meta.npy"))-128-64*3=1*2+2*7+4*4+8*4=64
        # hr=os.path.getsize(os.path.join(self.model_path, "slbrin_hrs.npy"))-128-64=hr_len*(1*2+2*1+8*6)=hr_len*52
        # model=os.path.getsize(os.path.join(self.model_path, "slbrin_models.npy"))-128=hr_len*model_size
        # cr=os.path.getsize(os.path.join(self.model_path, "slbrin_crs.npy"))-128-64=cr_len*(1*1+2*1+8*4)=cr_len*35
        # index_entries=os.path.getsize(os.path.join(self.model_path, "slbrin_data.npy"))-128
        # =hr_len*meta.threshold_number*(8*3+4)
        # Theoretically:
        # meta store last_hr/last_cr/5*ts/L=4+4+5*2+1=19
        # hr store value/length/number/*model/state=hr_len*(8+1+2+4+1)=hr_len*16
        # cr store value/number/state=cr_len*(8*4+2+1)=cr_len*35
        # index_entries is data_len*(8*3+4)=data_len*28
        data_len = sum([hr.number for hr in self.history_ranges]) + sum([cr.number for cr in self.current_ranges])
        hr_len = self.meta.last_hr + 1
        cr_len = self.meta.last_cr + 1
        return 19 + \
               hr_len * 16 + \
               os.path.getsize(os.path.join(self.model_path, "slbrin_models.npy")) - 128 + \
               cr_len * 35, data_len * ITEM_SIZE

    def model_clear(self):
        """
        clear the models which are not the best
        """
        for i in range(self.meta.last_hr + 1):
            tmp_index = NN(self.model_path, str(i), [0], [0],
                           None, None, None, None, None, None, None, None, None)
            tmp_index.clean_not_best_model_file()


# for query
valid_position_funcs = [
    lambda reg, window: None,
    lambda reg, window:  # right
    window[3] >= reg.left,
    lambda reg, window:  # left
    window[2] <= reg.right,
    lambda reg, window:  # left-right
    window[2] <= reg.right and reg.left <= window[3],
    lambda reg, window:  # up
    window[1] >= reg.bottom,
    lambda reg, window:  # up-right
    window[3] >= reg.left and window[1] >= reg.bottom,
    lambda reg, window:  # up-left
    window[2] <= reg.right and window[1] >= reg.bottom,
    lambda reg, window:  # up-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[1] >= reg.bottom,
    lambda reg, window:  # bottom
    window[0] <= reg.up,
    lambda reg, window:  # bottom-right
    window[3] >= reg.left and window[0] <= reg.up,
    lambda reg, window:  # bottom-left
    window[2] <= reg.right and window[0] <= reg.up,
    lambda reg, window:  # bottom-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[0] <= reg.up,
    lambda reg, window:  # bottom-up
    window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-right
    window[3] >= reg.left and reg.right and window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-left
    window[2] <= reg.right and window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[0] <= reg.up and reg.bottom <= window[1]]
range_position_funcs = [
    lambda reg, window, gh1, gh2, geohash: (None, None, None),
    lambda reg, window, gh1, gh2, geohash: (  # right
        None,
        geohash.encode(window[3], reg.up),
        lambda x: window[3] >= x[0]),
    lambda reg, window, gh1, gh2, geohash: (  # left
        geohash.encode(window[2], reg.bottom),
        None,
        lambda x: window[2] <= x[0]),
    lambda reg, window, gh1, gh2, geohash: (  # left-right
        geohash.encode(window[2], reg.bottom),
        geohash.encode(window[3], reg.up),
        lambda x: window[2] <= x[0] <= window[3]),
    lambda reg, window, gh1, gh2, geohash: (  # up
        None,
        geohash.encode(reg.right, window[1]),
        lambda x: window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-right
        None,
        gh2,
        lambda x: window[3] >= x[0] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-left
        geohash.encode(window[2], reg.bottom),
        geohash.encode(reg.right, window[1]),
        lambda x: window[2] <= x[0] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-left-right
        geohash.encode(window[2], reg.bottom),
        gh2,
        lambda x: window[2] <= x[0] <= window[3] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom
        geohash.encode(reg.left, window[0]),
        None,
        lambda x: window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-right
        geohash.encode(reg.left, window[0]),
        geohash.encode(window[3], reg.up),
        lambda x: window[3] >= x[0] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-left
        gh1,
        None,
        lambda x: window[2] <= x[0] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-left-right
        gh1,
        geohash.encode(window[3], reg.up),
        lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up
        geohash.encode(reg.left, window[0]),
        geohash.encode(reg.right, window[1]),
        lambda x: window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-right
        geohash.encode(reg.left, window[0]),
        gh2,
        lambda x: window[3] >= x[0] and window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-left
        gh1,
        geohash.encode(reg.right, window[1]),
        lambda x: window[2] <= x[0] and window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-left-right
        gh1,
        gh2,
        lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1] <= window[1])]


# for train
def build_nn(model_path, model_key, inputs, labels, is_new, is_simple, weight, core, train_step, batch_size,
             learning_rate, use_threshold, threshold, retrain_time_limit, tmp_dict):
    if is_simple:
        tmp_index = NNSimple(inputs, labels, weight, core, train_step, batch_size, learning_rate)
    else:
        tmp_index = NN(model_path, str(model_key), inputs, labels, is_new, weight, core,
                       train_step, batch_size, learning_rate, use_threshold, threshold, retrain_time_limit)
    tmp_index.build()
    abstract_index = AbstractNN(tmp_index.matrices, len(core) - 1,
                                math.floor(tmp_index.min_err),
                                math.ceil(tmp_index.max_err))
    del tmp_index
    gc.collect(generation=0)
    tmp_dict[model_key] = abstract_index


class Meta:
    def __init__(self, last_hr, last_cr, threshold_number, threshold_length, threshold_err, threshold_summary,
                 threshold_merge, geohash):
        # BRIN
        # SLBRIN
        self.last_hr = last_hr
        self.last_cr = last_cr
        self.threshold_number = threshold_number
        self.threshold_length = threshold_length
        self.threshold_err = threshold_err
        self.threshold_summary = threshold_summary
        self.threshold_merge = threshold_merge
        # self.L = L  # geohash.sum_bits
        # For compute
        self.geohash = geohash


class HistoryRange:
    def __init__(self, value, length, number, model, state, scope, value_diff):
        # BRIN
        self.value = value
        # SLBRIN
        self.length = length
        self.number = number
        self.model = model
        self.state = state
        # For compute
        self.scope = scope
        self.value_diff = value_diff
        self.max_key = number - 1

    def model_predict(self, x):
        x = self.model.predict((x - self.value) / self.value_diff - 0.5)
        if x <= 0:
            return 0
        elif x >= 1:
            return self.max_key
        return int(self.max_key * x)

    def update_error_range(self, xs):
        if self.number:
            # The amount of data is too much and predict is slow,
            # so uniform sampling is used to get 100 points to calculate the error
            if self.number > 100:
                step_size = self.number // 100
                sample_keys = [i for i in range(0, step_size * 100, step_size)]
                xs = np.array([[xs[i][2]] for i in sample_keys])
                ys = np.array(sample_keys)
            else:
                xs = np.array([[x[2]] for x in xs])
                ys = np.arange(self.number)
            # Optimization: single predict->collective predict: time ratio is 19:1
            pres = self.model.predicts((xs - self.value) / self.value_diff - 0.5)
            pres[pres < 0] = 0
            pres[pres > 1] = 1
            errs = pres * self.max_key - ys
            self.model.min_err = math.floor(errs.min())
            self.model.max_err = math.ceil(errs.max())
        else:
            self.model.min_err = 0
            self.model.max_err = 0


class CurrentRange:
    def __init__(self, value, number, state):
        # BRIN
        self.value = value
        # SLBRIN
        self.number = number
        self.state = state
        # For compute


class NN(MLP):
    def __init__(self, model_path, model_key, train_x, train_y, is_new, weight, core, train_step, batch_size,
                 learning_rate, use_threshold, threshold, retrain_time_limit):
        self.name = "SLBRIN NN"
        # The train_x's are ordered and the normalization does not need to compute the max-min values
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
        super().__init__(model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_new, weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                         retrain_time_limit)

    # Calculating err without considering breakpoints
    def get_err(self):
        inputs = self.train_x[1:-1]
        input_len = len(inputs)
        if input_len:
            pres = self.model(inputs).numpy().flatten()
            pres[pres < 0] = 0
            pres[pres > 1] = 1
            errs = pres * (input_len - 1) - np.arange(input_len)
            return errs.min(), errs.max()
        else:
            return 0.0, 0.0


class NNSimple(MLPSimple):
    def __init__(self, train_x, train_y, weight, core, train_step, batch_size, learning_rate):
        self.name = "SLBRIN NN"
        # The train_x's are ordered and the normalization does not need to compute the max-min values
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
        super().__init__(train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         weight, core, train_step, batch_size, learning_rate)

    # Calculating err without considering breakpoints
    def get_err(self):
        inputs = self.train_x[1:-1]
        input_len = len(inputs)
        if input_len:
            pres = self.model(inputs).numpy().flatten()
            pres[pres < 0] = 0
            pres[pres > 1] = 1
            errs = pres * (input_len - 1) - np.arange(input_len)
            return errs.min(), errs.max()
        else:
            return 0.0, 0.0


class AbstractNN:
    def __init__(self, matrices, hl_nums, min_err, max_err):
        self.matrices = matrices
        self.hl_nums = hl_nums
        self.min_err = min_err
        self.max_err = max_err

    # model.predict has a small deviation, probably the e of exp and the e of elu don't agree
    def predict(self, x):
        for i in range(self.hl_nums):
            x = relu(np.dot(x, self.matrices[i * 2]) + self.matrices[i * 2 + 1])
        return (np.dot(x, self.matrices[-2]) + self.matrices[-1])[0, 0]

    def predicts(self, xs):
        for i in range(self.hl_nums):
            xs = relu(np.dot(xs, self.matrices[i * 2]) + self.matrices[i * 2 + 1])
        return (np.dot(xs, self.matrices[-2]) + self.matrices[-1]).flatten()

    def splits(self):
        """
        Cut the matrix into quarters according to the input, limited to hidden layers = 1
        :return:
        """
        w0, b0, w1, b1 = self.matrices
        xbks = [[-0.5], [-0.25], [0], [0.25], [0.5]]
        ybks = self.predicts(xbks)
        m_0 = 0.25 * w0
        m01 = (-0.375 * w0 + b0).flatten()
        # The hidden layer w is (1, 128), b is (128,) and the calculated shape becomes (1, 128),
        # so it needs to be flattened to (128,)
        m02 = w1 / (ybks[1] - ybks[0])
        m03 = (b1 - ybks[0]) / (ybks[1] - ybks[0])
        m11 = (-0.125 * w0 + b0).flatten()
        m12 = w1 / (ybks[2] - ybks[1])
        m13 = (b1 - ybks[1]) / (ybks[2] - ybks[1])
        m21 = (0.125 * w0 + b0).flatten()
        m22 = w1 / (ybks[3] - ybks[2])
        m23 = (b1 - ybks[2]) / (ybks[3] - ybks[2])
        m31 = (0.375 * w0 + b0).flatten()
        m32 = w1 / (ybks[4] - ybks[3])
        m33 = (b1 - ybks[3]) / (ybks[4] - ybks[3])
        return [AbstractNN([m_0, m01, m02, m03], self.hl_nums, 0, 0),
                AbstractNN([m_0, m11, m12, m13], self.hl_nums, 0, 0),
                AbstractNN([m_0, m21, m22, m23], self.hl_nums, 0, 0),
                AbstractNN([m_0, m31, m32, m33], self.hl_nums, 0, 0)]


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/slbrin_skew/"
    data_distribution = Distribution.SKEW_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = SLBRIN(model_path=model_path)
    index_name = index.name
    load_index_from_file = False
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        # pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y/g)=8, slbrin global contiguous storage,
        # meta is 1 page, br paged store，model(2009)
        # hr size = value/length/number=16，1 page store 256 hr
        # cr size = value/number=35，1 page store 117 cr
        # model size =2009，1 page store 2 model
        # data size =x/y/g/key=8*3+4=28，1 page store 146 data
        # 10w，[1000]：289 cr
        # 1meta page，289/256=2hr page，1cr page, 289/2=145model page，10w/146=685data page
        # Single scan IO = read slbrin + read corresponding mod + read index item of model = 1+1+ error range/146/256
        # Index size = meta+hrs+crs+model+indexed items
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    threshold_number=10000,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    threshold_err=1,
                    threshold_summary=1000,
                    threshold_merge=5,
                    is_new=True,
                    is_simple=False,
                    weight=1,
                    core=[1, 128],
                    train_step=5000,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=1,
                    thread_pool_size=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    io_cost = 0
    model_num = index.meta.last_hr + 1
    logging.info("Model num: %s" % model_num)
    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
    model_precisions_avg = sum(model_precisions) / model_num
    logging.info("Error bound: %s" % model_precisions_avg)
    point_query_list = load_query(data_distribution, 0).tolist()
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
    logging.info("Sum up full cr time: %s" % index.sum_up_full_cr_time)
    logging.info("Merge outdated cr time: %s" % index.merge_outdated_cr_time)
    logging.info("Retrain inefficient model time: %s" % index.retrain_inefficient_model_time)
    logging.info("Retrain inefficient model num: %s" % index.retrain_inefficient_model_num)
    model_num = index.meta.last_hr + 1
    logging.info("Model num: %s" % model_num)
    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
    model_precisions_avg = sum(model_precisions) / model_num
    logging.info("Error bound: %s" % model_precisions_avg)
    update_time = end_time - start_time - \
                  index.sum_up_full_cr_time - index.merge_outdated_cr_time - index.retrain_inefficient_model_time
    logging.info("Update time: %s" % update_time)
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
