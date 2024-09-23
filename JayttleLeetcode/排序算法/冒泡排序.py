def bubbleSort(arr):
    # 检查输入数组是否有效
    if arr is None or len(arr) < 2:
        return arr

    n = len(arr)
    # 外层循环，控制需要进行的遍历次数
    for i in range(n):
        # 设置一个标志，判断本趟排序是否有交换
        swapped = False
        
        # 内层循环，进行相邻元素的比较与交换
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # 交换元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # 如果没有交换，说明数组已经有序，可以提前结束
        if not swapped:
            break
    
    return arr

# 示例使用
if __name__ == '__main__':
    example_array = [64, 34, 25, 12, 22, 11, 90]
    sorted_array = bubbleSort(example_array)
    print("排序后的数组:", sorted_array)