def halfsearch(arr, target):
    start = 0
    end = len(arr) - 1
    
    while start <= end:  # 当 start <= end 时继续循环
        mid = (start + end) // 2
        
        if arr[mid] == target:
            return mid  # 找到目标值，返回索引
        elif arr[mid] < target:
            start = mid + 1  # 目标值在右侧部分，更新 start
        else:
            end = mid - 1  # 目标值在左侧部分，更新 end

    return -1  # 目标值不在数组中，返回 -1

# 示例使用
if __name__ == '__main__':
    example_array = [1, 3, 5, 7, 9, 11, 13]
    target = 3
    index = halfsearch(example_array, target)
    print(f"目标值 {target} 的索引是: {index}")  # 输出: 目标值 7 的索引是: 3
