class Node:
    def __init__(self, value = None, left = None, right = None) -> None:
        self.value: int = value
        self.left: Node = left
        self.right: Node = right

def posOrderCur(node: Node):
    if node is None:
        return
    posOrderCur(node.left)
    posOrderCur(node.right)
    print(node.value, end=' ')

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
    posOrderCur(root) 