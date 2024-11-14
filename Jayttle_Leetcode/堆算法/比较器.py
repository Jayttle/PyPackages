class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __lt__(self, other):
        return self.age < other.age

    def __le__(self, other):
        return self.age <= other.age

    def __gt__(self, other):
        return self.age > other.age

    def __ge__(self, other):
        return self.age >= other.age

    def __eq__(self, other):
        return self.age == other.age

    def __ne__(self, other):
        return self.age != other.age

    def __repr__(self):
        return f'{self.name}({self.age})'

# 创建一些 Person 实例
alice = Person("Alice", 30)
bob = Person("Bob", 25)

# 使用比较器进行比较
print(alice > bob)  # True
print(alice < bob)  # False
print(alice >= bob) # True
print(alice <= bob) # False
print(alice == bob) # False
print(alice != bob) # True