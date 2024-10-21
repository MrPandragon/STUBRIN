import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def compute_pow(num):
    """
    计算幂
    """
    return math.log(num, 10)


def check_less_1(l):
    for nums in l:
        for num in nums[1]:
            if num < 1:
                return True
    return False


def plot_group_histogram(x, y_list, model_path, x_title, y_title, color_list, is_legend=True, legend_pos='best'):
    """
    分组直方图，y轴表示为10的幂次
    :param x: 组名list
    :param y_list: 组对应的值list
    :param model_path: 保存路径
    :param x_title: x轴标题
    :param y_title: y轴标题
    :param color_list: 组内各柱的颜色
    :param is_legend: 是否需要图例
    :param legend_pos: legend的位置
    """
    font_size = 20
    x_arange = np.arange(len(x))
    group_member_len = len(y_list)
    width = 0.1
    for i in range(group_member_len):
        plt.bar(x=x_arange - width * group_member_len / 2 + width / 2 * (i * 2 + 1),
                height=y_list[i][1],
                width=width,
                label=y_list[i][0],
                color=color_list[i])
    plt.ylabel(y_title, fontsize=font_size)
    plt.xlabel(x_title, fontsize=font_size)
    plt.xticks(x_arange, x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.yscale("log")
    plt.gcf().subplots_adjust(right=0.97, left=0.18, top=0.97, bottom=0.18)
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, fontsize=font_size, ncol=2, columnspacing=0.5)
    plt.savefig(model_path)
    plt.close()




def plot_result(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitors = ['RT', 'PRQT', 'BRINS', 'LISA', 'STUM', 'STUBRIN_NPS', 'STURBRIN']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#86B2C5', '#FCE166', '#F53935', '#B71C1C']
    # datasets = ['UNIFORM', 'NORMAL', 'SKEW', 'NYCT']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitors)
    dataset_len = len(datasets)

    # build time
    construction_times = [[competitors[j], [result.iloc[i * 9][j] for i in range(dataset_len)]]
                          for j in range(competitor_len)]
    plot_group_histogram(datasets, construction_times, output_path + "/build_time.png",
                         'Data distribution', 'Construction time (s)', competitor_colors, False)
    # index size
    index_structure_sizes = [[competitors[j],
                              [result.iloc[1 + i * 9][j] for i in range(dataset_len)]]
                             for j in range(competitor_len)]
    plot_group_histogram(datasets, index_structure_sizes, output_path + "/index_size.png",
                         'Data distribution', 'Index size (MB)', competitor_colors)

    # point query
    # io cost
    io_costs = [[competitors[j], [result.iloc[2 + i * 9][j] for i in range(dataset_len)]]
                for j in range(competitor_len)]
    plot_group_histogram(datasets, io_costs, output_path + "/io_cost.png",
                         'Data distribution', 'IO cost', competitor_colors, False)
    # query time
    point_query = [[competitors[j], [result.iloc[3 + i * 9][j] for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query, output_path + "/point_query.png",
                         'Data distribution', 'Average query time (μs)', competitor_colors)

    # range query
    # io cost
    io_costs = [[competitors[j], [result.iloc[4 + i * 9][j] for i in range(dataset_len)]]
                for j in range(competitor_len)]
    plot_group_histogram(datasets, io_costs, output_path + "/io_cost_range.png",
                         'Data distribution', 'IO cost', competitor_colors, False)
    # query time
    range_query = [[competitors[j], [result.iloc[5 + i * 9][j] for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query, output_path + "/range_query.png",
                         'Data distribution', 'Average query time (μs)', competitor_colors, True, 'upper left')

    # knn query
    # io cost
    io_costs = [[competitors[j], [result.iloc[6 + i * 9][j] for i in range(dataset_len)]]
                for j in range(competitor_len)]
    plot_group_histogram(datasets, io_costs, output_path + "/io_cost_knn.png",
                         'Data distribution', 'IO cost', competitor_colors, False)
    knn_query = [[competitors[j], [result.iloc[7 + i * 9][j] for i in range(dataset_len)]]
                 for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query, output_path + "/knn_query.png",
                         'Data distribution', 'Average query time (μs)', competitor_colors)

    # update time
    update_times = [[competitors[j], [result.iloc[8 + i * 9][j] for i in range(dataset_len)]]
                    for j in range(competitor_len)]
    plot_group_histogram(datasets, update_times, output_path + "/update_time.png",
                         'Data distribution', 'Update time (μs)', competitor_colors, False)



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    input_path = "./table/result.xlsx"
    output_path = "./png"
    plot_result(input_path, output_path)