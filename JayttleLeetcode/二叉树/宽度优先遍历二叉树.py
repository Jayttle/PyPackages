from collections import deque

class Node:
    def __init__(self, value = None, l = None, r = None) -> None:
        self.value: int = value  # 节点的值
        self.left: Node = l  # 左子节点
        self.right: Node = r  # 右子节点

    def __le__(self, other):
        return self.value <= other.value

def w(head: Node):
    if head is None:
        return  # 如果树为空，直接返回

    queue = deque([head])  # 初始化队列，将根节点加入队列
    while queue:
        node = queue.popleft()  # 从队列前端取出节点
        print(node.value, end=' ')  # 打印当前节点的值
        
        if node.left:
            queue.append(node.left)  # 如果有左子节点，将其加入队列
        if node.right:
            queue.append(node.right)  # 如果有右子节点，将其加入队列

# 示例用法
# 测试
if __name__ == "__main__":
    # 创建一棵树
    #     1
    #    / \
    #   2   3
    #  / \
    # 4   5
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    
    print("Pre-order Traversal (Recursive):")
    w(root) 