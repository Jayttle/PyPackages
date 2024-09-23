class Node:
    def __init__(self, value, left=None, right=None):
        self.value: int = value
        self.left: Node = left
        self.right: Node = right

def preOrderRecur(node: Node):
    if node is None:
        return
    print(node.value, end=' ')
    preOrderRecur(node.left)
    preOrderRecur(node.right)

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
    preOrderRecur(root)  # 输出: 1 2 4 5 3
