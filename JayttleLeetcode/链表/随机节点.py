class Node:
    def __init__(self, value, next=None, random=None) -> None:
        self.value: int = value
        self.next: Node = next
        self.random: Node = random

def copyRandomList(head: Node) -> Node:
    if head is None:
        return None

    # 1. 插入新节点
    current = head
    while current:
        new_node = Node(current.value, current.next)
        current.next = new_node
        current = new_node.next
    
    # 2. 复制随机指针
    current = head
    while current:
        new_node = current.next
        if current.random:
            new_node.random = current.random.next
        current = new_node.next
    
    # 3. 拆分链表
    old_list = head
    new_list = head.next
    new_head = new_list
    while old_list:
        old_list.next = old_list.next.next
        if new_list.next:
            new_list.next = new_list.next.next
        old_list = old_list.next
        new_list = new_list.next
    
    return new_head

# 测试
def print_list(head: Node):
    nodes = []
    while head:
        random_val = head.random.value if head.random else None
        nodes.append(f'{head.value} (Random: {random_val})')
        head = head.next
    print(' -> '.join(nodes))

if __name__ == "__main__":
    # 创建链表: 7 -> 13 -> 11 -> 10 -> 1
    n1 = Node(7)
    n2 = Node(13)
    n3 = Node(11)
    n4 = Node(10)
    n5 = Node(1)

    n1.next = n2
    n2.next = n3
    n3.next = n4
    n4.next = n5

    n1.random = None
    n2.random = n1
    n3.random = n5
    n4.random = n3
    n5.random = n1

    copied_head = copyRandomList(n1)
    print("Original list:")
    print_list(n1)
    print("Copied list:")
    print_list(copied_head)
