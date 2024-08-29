def printAllFoldes(n):
    printProcess(1, n, True)

def printProcess(i, n, down):
    if i > n:
        return
    printProcess(i + 1, n, True)
    # 使用 Python 的条件表达式来进行布尔值判断
    print('凹' if down else '凸')
    printProcess(i + 1, n, False)

if __name__ == '__main__':
    n = 3
    printAllFoldes(n)
