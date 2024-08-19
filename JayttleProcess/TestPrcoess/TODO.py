#TODO: 比较双天线姿态与倾斜仪数据的对比
import math
import matplotlib.pyplot as plt
import numpy as np
# # 第一个点的直角坐标
# x1 = 594771.8212410026
# y1 = 3927701.7185229557

# # 第二个点的直角坐标
# x2 = 594771.9791162078
# y2 = 3927704.6989034447

# ggkxR071-avg_east, avg_north:594771.8212410026,3927701.7185229557
# ggkxR072-avg_east, avg_north:594771.9791162078,3927704.6989034447
# 第一个点的直角坐标
import numpy as np

# Example correlation matrix (replace with your actual matrix)
correlation_matrix = np.array([
    [ 1.00000000e+00,  8.79711872e-03, -1.18068024e-02, -2.77363747e-02],
    [ 8.79711872e-03,  1.00000000e+00, -3.31313479e-04,  1.04194177e-02],
    [-1.18068024e-02, -3.31313479e-04,  1.00000000e+00, -1.77534189e-01],
    [-2.77363747e-02,  1.04194177e-02, -1.77534189e-01,  1.00000000e+00]
])

# Set numpy's printing options to suppress scientific notation
np.set_printoptions(suppress=True)

# Print the correlation matrix without scientific notation
print("Correlation matrix:")
print(correlation_matrix)

