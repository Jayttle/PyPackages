def set_task_completed(task_id, bitfield):
    # 检查任务 ID 是否在有效范围内
    if task_id < 1 or task_id > 1024:
        return -1
    # 任务 ID 从 1 开始，转换为位操作索引
    bit_index = task_id - 1
    # 使用位操作设置对应位为 1（已完成）
    bitfield |= (1 << bit_index)
    return bitfield

def is_task_completed(task_id, bitfield):
    # 检查任务 ID 是否在有效范围内
    if task_id < 1 or task_id > 1024:
        return -1
    # 任务 ID 从 1 开始，转换为位操作索引
    bit_index = task_id - 1
    # 检查对应位是否为 1（已完成）
    return 1 if (bitfield & (1 << bit_index)) != 0 else 0

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # 读取任务 ID
    task1_id = int(data[0])
    task2_id = int(data[1])
    
    # 初始化 32 位整数（0 表示所有任务未完成）
    bitfield = 0
    
    # 设置第一个任务为已完成
    bitfield = set_task_completed(task1_id, bitfield)
    
    # 检查第二个任务是否已完成
    result = is_task_completed(task2_id, bitfield)
    
    # 输出结果
    print(result)

if __name__ == "__main__":
    main()
