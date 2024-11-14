class Node:
    def __init__(self, value, left=None, right=None) -> None:
        self.value = value
        self.left: Node = left
        self.right: Node = right

class ReturnType:
    def __init__(self, isbalanced, height) -> None:
        self.isBalanced: bool = isbalanced
        self.height: int = height

def isBalanced(head):
    return process(head).isBalanced

def process(node: Node) -> ReturnType:
    if node is None:
        return ReturnType(True, 0)
    leftData = process(node.left)
    rightData = process(node.right)
    height = max(leftData.height, rightData.height) + 1
    isBalanced = (leftData.isBalanced and rightData.isBalanced and abs(leftData.height - rightData.height) < 2)

        
    # 调试输出
    print(f"Node Value: {node.value}, Left Height: {leftData.height}, Right Height: {rightData.height}, Is Balanced: {isBalanced}")

    return ReturnType(isBalanced, height)

# 测试
if __name__ == "__main__":
        # 创建一个不是完全二叉树的树
    #     1
    #    / \
    #   2   3
    #  / \
    # 4   5
    #    /
    #   6
    root_invalid = Node(1)
    root_invalid.left = Node(2)
    root_invalid.right = Node(3)
    root_invalid.left.left = Node(4)
    root_invalid.left.right = Node(5)
    root_invalid.left.right.left = Node(6)

    # 检查树是否平衡
    print(isBalanced(root_invalid))  # 应该输出: False
