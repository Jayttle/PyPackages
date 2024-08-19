class Node:
    def __init__(self, value, next=None) -> None:
        self.value: int = value
        self.next: Node = next

def isPalindrome(head: Node) -> bool:
    if head is None or head.next is None:
        return True

    # 快慢指针找到链表中点
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

    # 反转链表的后半部分
    prev = None
    second_half = slow
    while second_half is not None:
        next_node = second_half.next
        second_half.next = prev
        prev = second_half
        second_half = next_node

    # 比较前半部分和反转后的后半部分
    first_half = head
    second_half = prev
    while second_half is not None:
        if first_half.value != second_half.value:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True

# 测试
if __name__ == "__main__":
    # 创建链表: 1 -> 2 -> 2 -> 1
    head = Node(1, Node(2, Node(2, Node(1))))
    print(isPalindrome(head))  # 输出: True

    # 创建链表: 1 -> 2 -> 3
    head = Node(1, Node(2, Node(3)))
    print(isPalindrome(head))  # 输出: False
