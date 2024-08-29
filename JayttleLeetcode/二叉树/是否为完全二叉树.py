from collections import deque

class Node:
    def __init__(self, value, left=None, right=None):
        self.value: int = value
        self.left: Node = left
        self.right: Node = right

def is_complete_binary_tree(root: Node) -> bool:
    if root is None:
        return True
    
    # 用于层次遍历的队列
    queue = deque([root])
    
    # 标记是否已遇到非完全节点
    found_non_full_node = False
    
    while queue:
        node = queue.popleft()
        
        if node.left:
            if found_non_full_node:
                return False
            queue.append(node.left)
        else:
            found_non_full_node = True
        
        if node.right:
            if found_non_full_node:
                return False
            queue.append(node.right)
        else:
            found_non_full_node = True
    
    return True

# 测试函数
if __name__ == "__main__":
    # 创建一棵完全二叉树
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
    
    print("Is Complete Binary Tree:", is_complete_binary_tree(root))  # 输出: True

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
    root_invalid.right.left = Node(6)
    
    print("Is Complete Binary Tree:", is_complete_binary_tree(root_invalid))  # 输出: False
