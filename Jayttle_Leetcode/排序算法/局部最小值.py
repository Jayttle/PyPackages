def partMin(arr):
    if arr is None or len(arr) == 0:
        return None  # 数组为空的情况下返回 None
    if len(arr) == 1:
        return arr[0]  # 数组只有一个元素时返回这个元素
    
    def findLocalMin(start, end):
        if start == end:
            return arr[start]
        
        mid = (start + end) // 2

        # 检查中间元素是否是局部最小值
        if (mid == 0 or arr[mid] < arr[mid - 1]) and (mid == len(arr) - 1 or arr[mid] < arr[mid + 1]):
            return arr[mid]
        elif mid < len(arr) - 1 and arr[mid] > arr[mid + 1]:
            # 如果中间元素大于右边的元素，局部最小值在右边
            return findLocalMin(mid + 1, end)
        else:
            # 如果中间元素大于左边的元素，局部最小值在左边
            return findLocalMin(start, mid - 1)
    
    return findLocalMin(0, len(arr) - 1)


if __name__ == '__main__':
    example_array = [9, 6, 4, 7, 3, 7, 8]
    local_min = partMin(example_array)
    print("局部最小值是:", local_min)
