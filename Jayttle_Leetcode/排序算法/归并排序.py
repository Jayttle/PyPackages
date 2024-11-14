def mergeSort(arr):
    if len(arr) <= 1:
        return arr  # 如果数组长度小于等于1，则不需要排序，直接返回
    return process(arr, 0, len(arr) - 1)

def process(arr, L, R):
    if L == R:
        return [arr[L]]  # 如果只有一个元素，直接返回该元素的列表
    
    mid = L + (R - L) // 2  # 计算中点
    leftarr = process(arr, L, mid)  # 递归处理左半部分
    rightarr = process(arr, mid + 1, R)  # 递归处理右半部分
    
    return merge(leftarr, rightarr)  # 合并左右部分

def merge(leftarr, rightarr):
    merged = []
    i, j = 0, 0
    
    # 合并两个已排序的数组
    while i < len(leftarr) and j < len(rightarr):
        if leftarr[i] <= rightarr[j]:
            merged.append(leftarr[i])
            i += 1
        else:
            merged.append(rightarr[j])
            j += 1
    
    # 如果左数组还有剩余元素
    if i < len(leftarr):
        merged.extend(leftarr[i:])
    
    # 如果右数组还有剩余元素
    if j < len(rightarr):
        merged.extend(rightarr[j:])
    
    return merged


if __name__ == '__main__':
    example_array = [3, 5, 1, 8, 2, 9, 4]
    sorted_array = mergeSort(example_array)
    print("排序后的数组是:", sorted_array)  # 输出: 排序后的数组是: [1, 2, 3, 4, 5, 8, 9]
