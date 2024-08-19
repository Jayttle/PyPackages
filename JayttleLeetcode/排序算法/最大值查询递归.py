def getMax(arr):
    if not arr:
        return None  # 如果数组为空，返回 None
    return process(arr, 0, len(arr) - 1)

def process(arr, L, R):
    if L == R:
        return arr[L]  # 如果只有一个元素，返回该元素
    mid = L + (R - L) // 2  # 正确计算中点
    leftMax = process(arr, L, mid)  # 递归查找左半部分的最大值
    rightMax = process(arr, mid + 1, R)  # 递归查找右半部分的最大值
    return max(leftMax, rightMax)  # 返回左右半部分中的最大值

if __name__ == '__main__':
    example_array = [3, 5, 1, 8, 2, 9, 4]
    max_value = getMax(example_array)
    print("数组中的最大值是:", max_value)  # 输出: