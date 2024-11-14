class numpair:
    def __init__(self, num1, num2) -> None:
        self.num1 = num1
        self.num2 = num2

    def __lt__(self, other):
        return self.num1 + self.num2 < other.num1 + other.num2

    def __str__(self) -> str:
        return f'[{self.num1}, {self.num2}]'

def heapify(arr, index, heapsize):
    left = index * 2 + 1
    right = index * 2 + 2
    minidx = index
    if left < heapsize and arr[left] < arr[minidx]:
        minidx = left
    if right < heapsize and arr[right] < arr[minidx]:
        minidx = right
    if minidx != index:
        arr[index], arr[minidx] = arr[minidx], arr[index]
        heapify(arr, minidx, heapsize)

def heapInsert(arr, index):
    while index > 0 and arr[(index - 1) // 2] > arr[index]:
        parent = (index - 1) // 2
        arr[parent], arr[index] = arr[index], arr[parent]
        index = parent

def buildHeap(arr):
    heapsize = len(arr)
    for i in range(heapsize // 2 - 1, -1, -1):
        heapify(arr, i, heapsize)

def removeMin(arr):
    if not arr:
        return None
    minElem = arr[0]
    arr[0] = arr[-1]
    arr.pop()  # Remove the last element
    heapify(arr, 0, len(arr))
    return minElem

if __name__ == '__main__':
    arr1 = [1, 7, 11]
    arr2 = [2, 4, 6]
    arr = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            arr.append(numpair(arr1[i], arr2[j]))
    buildHeap(arr)
    k = 3
    res_arr = []
    for _ in range(k):
        res_arr.append(removeMin(arr))
    for item in res_arr:
        print(item, end=' ')
