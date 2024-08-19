def helansort(arr, num):
    startidx = 0
    for i in range(len(arr)):
        if arr[i] <= num:
            arr[startidx], arr[i] = arr[i], arr[startidx]
            startidx += 1

def helansort2(arr, num):
    startidx = 0
    endidx = len(arr) - 1
    i = 0

    while i <= endidx:
        if arr[i] < num:
            arr[startidx], arr[i] = arr[i], arr[startidx]
            startidx += 1
            i += 1
        elif arr[i] > num:
            arr[endidx], arr[i] = arr[i], arr[endidx]
            endidx -= 1
        else:  # arr[i] == num
            i += 1

if __name__ == '__main__':
    arr = [3, 5, 6, 3, 4, 5, 2, 6, 9, 0]
    helansort2(arr, 5)
    print(f'最终结果 = {arr}')