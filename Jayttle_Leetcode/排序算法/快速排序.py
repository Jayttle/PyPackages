def quick_sort_in_place(arr, low, high):
    """
    原地快速排序
    :param arr: 待排序的数组
    :param low: 当前处理的子数组的起始索引
    :param high: 当前处理的子数组的结束索引
    """
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_in_place(arr, low, pi - 1)
        quick_sort_in_place(arr, pi + 1, high)

def partition(arr, low, high):
    """
    分区操作，选择基准点并将数组分成两个部分
    :param arr: 待排序的数组
    :param low: 当前处理的子数组的起始索引
    :param high: 当前处理的子数组的结束索引
    :return: 基准点的最终位置
    """
    pivot = arr[high]  # 选择最后一个元素作为基准点
    i = low - 1  # i 指向小于基准点的部分的最后一个元素

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 测试原地快速排序
if __name__ == "__main__":
    array = [3, 6, 8, 10, 1, 2, 1]
    print("原始数组:", array)
    quick_sort_in_place(array, 0, len(array) - 1)
    print("排序后的数组:", array)