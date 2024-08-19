def heapify(arr, index, heapSize):
    # 计算当前节点的左右子节点的索引
    left = 2 * index + 1
    right = 2 * index + 2
    smallest = index  # 假设当前节点是最小的

    # 如果左子节点存在且小于当前节点，则更新 smallest
    if left < heapSize and arr[left] < arr[smallest]:
        smallest = left

    # 如果右子节点存在且小于当前节点，则更新 smallest
    if right < heapSize and arr[right] < arr[smallest]:
        smallest = right

    # 如果 smallest 不是当前节点，则交换当前节点与 smallest 节点的值
    if smallest != index:
        arr[index], arr[smallest] = arr[smallest], arr[index]
        # 递归调用 heapify 以保证调整后的子树满足小根堆的性质
        heapify(arr, smallest, heapSize)

def heapInsert(arr, index):
    # 上浮操作，保持小根堆的性质
    while index > 0 and arr[index] < arr[(index - 1) // 2]:
        parent = (index - 1) // 2
        arr[index], arr[parent] = arr[parent], arr[index]
        index = parent

def buildHeap(arr):
    # 从数组中构建小根堆
    heapSize = len(arr)
    # 从最后一个非叶子节点开始，逐步调整堆
    for i in range(heapSize // 2 - 1, -1, -1):
        heapify(arr, i, heapSize)

def heapSort(arr):
    # 构建小根堆
    buildHeap(arr)
    heapSize = len(arr)

    # 进行堆排序
    for i in range(heapSize - 1, 0, -1):
        # 将堆顶元素（最小值）交换到数组的末尾
        arr[0], arr[i] = arr[i], arr[0]
        # 调整堆，使剩余的部分重新成为小根堆
        heapify(arr, 0, i)

def updateValue(arr, i, newValue):
    # 更新第 i 个位置的值
    oldValue = arr[i]
    arr[i] = newValue
    
    # 如果新值小于旧值，进行上浮操作
    if newValue < oldValue:
        heapInsert(arr, i)
    # 如果新值大于旧值，进行下沉操作
    elif newValue > oldValue:
        heapify(arr, i, len(arr))

if __name__ == '__main__':
    arr = [3, 5, 6, 3, 4, 5, 2, 6, 9, 0]
    
    # 构建小根堆
    buildHeap(arr)
    print(f'构建的小根堆 = {arr}')
    
    # 更新第 i 位置的值
    i = 4  # 需要更新的位置
    newValue = 1  # 新值
    updateValue(arr, i, newValue)
    print(f'更新第 {i} 位置后的堆 = {arr}')
    
    # 进行堆排序
    heapSort(arr)
    print(f'排序后的数组 = {arr}')
