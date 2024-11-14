def selectionSort(arr):
    # 检查输入是否有效
    if arr is None or len(arr) < 2:
        return

    # 外层循环遍历整个数组
    for i in range(len(arr)):
        minIndex = i  # 假设当前位置是最小值的索引
        
        # 内层循环找到最小值的索引
        for j in range(i + 1, len(arr)): 
            if arr[j] < arr[minIndex]:  # 如果找到更小的值
                minIndex = j  # 更新最小值的索引
        
        # 交换当前位置和找到的最小值位置
        if minIndex != i:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]

if __name__ == '__main__':
    arr = [64, 25, 12, 22, 11]
    selectionSort(arr)
    print("Sorted array:", arr)