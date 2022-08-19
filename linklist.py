class Node:
    def __init__(self, name):
        self.name = name
        self.next = None


class Linklist:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head == None

    def length(self):
        '''
        链表长度
        :return: count -> Int
        '''
        # cur游标， 用来移动遍历节点
        cur = self.head
        # count记录数量
        count = 1
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        '''
        遍历整个链表
        :return: None
        '''
        cur = self.head
        while cur != None:
            print(cur.name)
            cur = cur.next

    def add(self, item):
        '''
        链表尾部添加元素
        '''
        node = Node(item)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def remove(self, item):
        """删除节点"""
        cur = self.head

        # 如果删除的节点是头节点
        if cur.name == item:
            self.head = cur.next
        else:
            # 找到删除节点的前驱节点
            while cur.next.name != item:
                cur = cur.next
            cur.next = cur.next.next