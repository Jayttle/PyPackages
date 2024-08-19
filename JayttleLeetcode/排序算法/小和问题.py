def smallSum(arr):
    if arr is None or len(arr) < 2:
        return 0
    
    def process(arr, l, r):
        print(f"处理范围 l={l}, r={r}")  # 打印当前处理的范围
        if l >= r:
            return 0
        mid = l + (r - l) // 2
        left_sum = process(arr, l, mid)
        right_sum = process(arr, mid + 1, r)
        merge_sum = merge(arr, l, mid, r)
        print(f"处理范围 l={l}, r={r} -> 左侧和={left_sum}, 右侧和={right_sum}, 合并和={merge_sum}")
        return left_sum + right_sum + merge_sum

    def merge(arr, l, m, r):
        print(f"合并范围 l={l}, m={m}, r={r}")  # 打印当前合并的范围
        help_arr = []
        p1 = l
        p2 = m + 1
        res = 0

        while p1 <= m and p2 <= r:
            if arr[p1] < arr[p2]:
                res += (r - p2 + 1) * arr[p1]
                help_arr.append(arr[p1])
                print(f"从左半部分添加 {arr[p1]}，当前合并和={res}")
                p1 += 1
            else:
                help_arr.append(arr[p2])
                print(f"从右半部分添加 {arr[p2]}")
                p2 += 1

        while p1 <= m:
            help_arr.append(arr[p1])
            print(f"添加左半部分剩余元素 {arr[p1]}")
            p1 += 1

        while p2 <= r:
            help_arr.append(arr[p2])
            print(f"添加右半部分剩余元素 {arr[p2]}")
            p2 += 1

        arr[l:r+1] = help_arr
        print(f"合并后数组区间 arr[{l}:{r+1}] = {arr[l:r+1]}")
        return res

    return process(arr, 0, len(arr) - 1)

if __name__ == '__main__':
    arr = [4, 3, 5, 7, 1, 6] 
    res = smallSum(arr)
    print(f'最终结果 = {res}')