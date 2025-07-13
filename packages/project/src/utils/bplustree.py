from collections import OrderedDict
import typing as t

T = t.TypeVar("T")


class BPlusTreeNode(t.Generic[T]):
    def __init__(self, is_leaf: bool = True):
        self.keys: list[int] = []
        self.values: list[T] = []
        self.children: list[BPlusTreeNode[T]] = []
        self.is_leaf: bool = is_leaf
        self.parent: BPlusTreeNode[T] | None = None
        self.next: BPlusTreeNode[T] | None = None


class BPlusTree(t.Generic[T]):
    def __init__(self, order: int = 3, max_size: int = 128):
        self.root = BPlusTreeNode[T](is_leaf=True)
        self.order = order
        self.max_size: int = max_size
        self.size = 0

        self._lru = OrderedDict[int, None]()

    def get(self, key: int) -> T | None:
        leaf = self._find_leaf(self.root, key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                self._lru_move_to_end(key)
                return leaf.values[i]
        return None

    def insert(self, key: int, value: T):

        leaf = self._find_leaf(self.root, key)
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.values[i] = value
                self._lru_move_to_end(key)
                return

        if self.max_size is not None and self.size >= self.max_size:
            self._evict_lru()

        self._insert_into_leaf(leaf, key, value)
        self.size += 1
        self._lru[key] = None
        self._lru_move_to_end(key)

    def delete(self, key: int):
        leaf = self._find_leaf(self.root, key)
        if key not in leaf.keys:
            return
        idx = leaf.keys.index(key)
        del leaf.keys[idx]
        del leaf.values[idx]
        self.size -= 1
        self._lru.pop(key, None)
        self._rebalance_after_delete(leaf)

    def _evict_lru(self):
        if not self._lru:
            return
        lru_key, _ = self._lru.popitem(last=False)
        self.delete(lru_key)

    def _lru_move_to_end(self, key: int):
        """Marks the key as recently used."""
        if key in self._lru:
            self._lru.move_to_end(key)

    def _find_leaf(self, node: BPlusTreeNode[T], key: int) -> BPlusTreeNode[T]:
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    def _insert_into_leaf(self, leaf: BPlusTreeNode[T], key: int, value: T):
        i = 0
        while i < len(leaf.keys) and key > leaf.keys[i]:
            i += 1
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)

        if len(leaf.keys) > self.order:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf: BPlusTreeNode[T]):
        mid = (len(leaf.keys) + 1) // 2
        new_leaf = BPlusTreeNode[T](is_leaf=True)
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        new_leaf.next = leaf.next
        leaf.next = new_leaf

        self._insert_into_parent(leaf, new_leaf.keys[0], new_leaf)

    def _insert_into_parent(
        self, node: BPlusTreeNode[T], key: int, new_node: BPlusTreeNode[T]
    ):
        if node is self.root:
            new_root = BPlusTreeNode[T](is_leaf=False)
            new_root.keys = [key]
            new_root.children = [node, new_node]
            self.root = new_root
            node.parent = new_root
            new_node.parent = new_root
            return

        parent = node.parent
        insert_idx = parent.children.index(node) + 1
        parent.children.insert(insert_idx, new_node)
        parent.keys.insert(insert_idx - 1, key)
        new_node.parent = parent

        if len(parent.keys) > self.order:
            self._split_internal(parent)

    def _split_internal(self, node: BPlusTreeNode[T]):
        mid = len(node.keys) // 2
        mid_key = node.keys[mid]

        new_node = BPlusTreeNode[T](is_leaf=False)
        new_node.keys = node.keys[mid + 1 :]
        new_node.children = node.children[mid + 1 :]

        for child in new_node.children:
            child.parent = new_node

        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]

        self._insert_into_parent(node, mid_key, new_node)

    def _rebalance_after_delete(self, node: BPlusTreeNode[T]):
        if node is self.root:
            if not node.is_leaf and len(node.children) == 1:
                self.root = node.children[0]
                self.root.parent = None
            return

        min_keys = (self.order + 1) // 2

        if len(node.keys) >= min_keys:
            return

        parent = node.parent
        index = parent.children.index(node)

        if index > 0:
            left = parent.children[index - 1]
            if len(left.keys) > min_keys:
                if node.is_leaf:
                    node.keys.insert(0, left.keys.pop())
                    node.values.insert(0, left.values.pop())
                    parent.keys[index - 1] = node.keys[0]
                else:
                    node.keys.insert(0, parent.keys[index - 1])
                    parent.keys[index - 1] = left.keys.pop()
                    node.children.insert(0, left.children.pop())
                    node.children[0].parent = node
                return

        if index < len(parent.children) - 1:
            right = parent.children[index + 1]
            if len(right.keys) > min_keys:
                if node.is_leaf:
                    node.keys.append(right.keys.pop(0))
                    node.values.append(right.values.pop(0))
                    parent.keys[index] = right.keys[0]
                else:
                    node.keys.append(parent.keys[index])
                    parent.keys[index] = right.keys.pop(0)
                    node.children.append(right.children.pop(0))
                    node.children[-1].parent = node
                return

        if index > 0:
            self._merge_nodes(parent.children[index - 1], node)
        else:
            self._merge_nodes(node, parent.children[index + 1])

    def _merge_nodes(self, left: BPlusTreeNode[T], right: BPlusTreeNode[T]):
        parent = left.parent
        index = parent.children.index(left)

        if left.is_leaf:
            left.keys.extend(right.keys)
            left.values.extend(right.values)
            left.next = right.next

            parent.keys.pop(index)
        else:
            separator = parent.keys.pop(index)
            left.keys.append(separator)
            left.keys.extend(right.keys)
            left.children.extend(right.children)
            for child in right.children:
                child.parent = left

        parent.children.remove(right)

        if parent is self.root and not parent.keys:
            self.root = left
            left.parent = None
        else:

            self._rebalance_after_delete(parent)
