def insertionSort(arr):
    if arr is None or len(arr) < 2:
        return
    for i in range(1, len(arr)):
         for j in range(i, 0, -1):
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
            else:
                break

def insertionSort2(arr):
    if arr is None or len(arr) < 2:
        return
    for i in range(1, len(arr)):
        key = arr[i]  # 当前要插入的元素
        j = i - 1
        
        # 将大于 key 的元素向后移动一位
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # 插入 key 到正确的位置
        arr[j + 1] = key

if __name__ == '__main__':
    arr = [64, 25, 12, 22, 11]
    insertionSort(arr)
    print("Sorted array:", arr)