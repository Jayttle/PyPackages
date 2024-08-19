class Node:
    def __init__(self, value, next=None) -> None:
        self.value: int = value
        self.next: Node = next

def isPalindrome(head: Node) -> bool:
    if head is None or head.next is None:
        return True

    # 使用栈来存储链表节点的值
    stack = []
    current = head
    
    # 将链表中的所有值推入栈
    while current is not None:
        stack.append(current.value)
        current = current.next
    
    # 再次遍历链表并与栈中的值进行比较
    current = head
    while current is not None:
        if stack.pop() != current.value:
            return False
        current = current.next

    return True

# 测试
if __name__ == "__main__":
    # 创建链表: 1 -> 2 -> 2 -> 1
    head = Node(1, Node(2, Node(2, Node(1))))
    print(isPalindrome(head))  # 输出: True

    # 创建链表: 1 -> 2 -> 3
    head = Node(1, Node(2, Node(3)))
    print(isPalindrome(head))  # 输出: False
