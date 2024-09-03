def quickSort(arr, l ,r):
    if l < r:
        pi = process(arr, l, r)
        quickSort(arr, l, pi - 1)
        quickSort(arr, pi + 1, r)