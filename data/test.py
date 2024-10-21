import numpy as np
import pandas
import matplotlib.pyplot as plt


def plot_scatter_xy(data):
    # Load data from the npy file
    data_list = np.array(data)

    # Extract x and y values
    x_values = data_list[:, 0]
    y_values = data_list[:, 1]

    # Create a scatter plot for x and y values
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, s=1, c='blue', alpha=0.6, edgecolors='w')
    plt.title('Scatter Plot of x and y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    # Show the plot
    plt.show()

def bar_position(x):
    # 假设有三组数据，每组有7个数据
    group_member_len = 7  # 每组有7个数据

    # 根据代码中的逻辑，柱状图的宽度是 0.1
    width = 0.1

    # 计算每条柱状图的宽度如何影响分布
    x_arange = np.arange(len(x))  # x轴上的组的位置

    # 列表保存每组数据柱状图的中心位置
    bar_positions = []
    for i in range(group_member_len):
        bar_position = x_arange - width * group_member_len / 2 + width / 2 * (i * 2 + 1)
        bar_positions.append(bar_position)

    return bar_positions


if __name__ == "__main__":
    # input_path = "./table/skew_1.npy"
    # read the data
    # data = np.load(input_path, allow_pickle=True).tolist()
    # # convert to pandas dataframe
    # df = pandas.DataFrame(data)
    # print 10 rows
    # print(df.head(10))
    #
    # plot_scatter_xy(data)

    print(bar_position(x=['Group 1', 'Group 2', 'Group 3']))
    print(bar_position(x=['Group 1', 'Group 2', 'Group 3', 'Group 4']))
