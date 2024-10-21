import os
import time

from src.experiment.common_utils import Distribution, load_data, data_precision, data_region
from src.utils.geohash_utils import Geohash

"""
Experimental: time to test multidimensional sorting alone
1. Convert dataset from %s to %s_SORTED
2. Subsequent experiments use %s_SORTED directly to build the index and add the time of multidimensional sorting to the build time
"""

os.chdir(os.path.dirname(os.path.realpath(__file__)))
data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
for data_distribution in data_distributions:
    data = load_data(data_distribution, 0)
    geohash = Geohash.init_by_precision(data_precision=data_precision[data_distribution],
                                        region=data_region[data_distribution])
    start_time = time.time()
    data = [(data[i][0], data[i][1], geohash.encode(data[i][0], data[i][1]), i) for i in range(len(data))]
    data.sort(key=lambda x: x[2])
    end_time = time.time()
    search_time = end_time - start_time
    print("Compute geohash time: %s" % search_time)
