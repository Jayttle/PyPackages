def p(i, from_, to_, other):
    if i == 1:
        print(f"Move disk 1 from {from_} to {to_}")
        return
    # 移动前 i-1 个盘子从 from_ 到 other
    p(i - 1, from_, other, to_)
    # 移动第 i 个盘子从 from_ 到 to_
    print(f"Move disk {i} from {from_} to {to_}")
    # 移动 i-1 个盘子从 other 到 to_
    p(i - 1, other, to_, from_)

# 示例用法
if __name__ == "__main__":
    num_disks = 3  # 可以修改为需要的盘子数量
    print(f"Solving Tower of Hanoi with {num_disks} disks:")
    p(num_disks, 'A', 'C', 'B')