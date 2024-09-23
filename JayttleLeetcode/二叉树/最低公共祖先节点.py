class Node:
    def __init__(self, value, left=None, right=None):
        self.value: int = value
        self.left: Node = left
        self.right: Node = right

def process(head: Node, fatherMap: dict):
    """建立每个节点的父节点映射"""
    if head is None:
        return
    if head.left:
        fatherMap[head.left] = head
        process(head.left, fatherMap)
    if head.right:
        fatherMap[head.right] = head
        process(head.right, fatherMap)

def lca(head: Node, o1: Node, o2: Node) -> Node:
    """
    找到二叉树中两个节点 o1 和 o2 的最低公共祖先。
    
    :param head: 树的根节点
    :param o1: 第一个目标节点
    :param o2: 第二个目标节点
    :return: 低级公共祖先节点
    """
    fatherMap = {}
    fatherMap[head] = None  # 根节点的父节点为 None
    process(head, fatherMap)
    
    # 创建一个集合来存储从 o1 向上到根的所有节点
    ancestors = set()
    
    # 从 o1 开始，逐步向上遍历到根节点
    cur = o1
    while cur is not None:
        ancestors.add(cur)
        cur = fatherMap.get(cur)
    
    # 从 o2 开始，逐步向上遍历并查找第一个在 ancestors 中的节点
    cur = o2
    while cur not in ancestors:
        cur = fatherMap.get(cur)
    
    return cur

# 测试用例
if __name__ == "__main__":
    # 创建一个示例树
    #         1
    #        / \
    #       2   3
    #      / \ / \
    #     4  5 6  7
    #    /
    #   8
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)
    root.left.right.left = Node(8)
    
    # 查询节点
    o1 = root.left.left  # 节点 4
    o2 = root.left.right.left  # 节点 8
    
    # 查询最低公共祖先
    ancestor = lca(root, o1, o2)
    if ancestor:
        print(f"The lowest common ancestor of nodes {o1.value} and {o2.value} is {ancestor.value}.")
    else:
        print("One or both nodes are not present in the tree.")
