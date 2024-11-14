def findOddOccurrence(arr):
    result = 0
    for num in arr:
        result ^= num  # 使用异或运算
    return result


def findTwoOddOccurrences(arr):
    # 第一步：异或所有元素，得到两个奇数次出现的数的异或结果
    xor_result = 0
    for num in arr:
        xor_result ^= num

    # 找到 xor_result 中最右侧的1（用于分组）
    diff_bit = xor_result & -xor_result

    # 使用 diff_bit 将数组分成两组，分别计算每组的异或结果
    num1, num2 = 0, 0
    for num in arr:
        if num & diff_bit:
            num1 ^= num
        else:
            num2 ^= num

    return num1, num2



# 示例使用
if __name__ == '__main__':
    example_array = [2, 3, 5, 4, 5, 3, 4, 4, 4]
    odd_occurrence_number = findOddOccurrence(example_array)
    print("出现奇数次的数字是:", odd_occurrence_number)

    # 示例使用
    example_array = [1, 2, 1, 3, 2, 5, 5, 6]
    odd_occurrences = findTwoOddOccurrences(example_array)
    print("出现奇数次的两个数字是:", odd_occurrences)