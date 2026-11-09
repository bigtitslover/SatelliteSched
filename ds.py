"""Core data structures for SatelliteSched.

Everything here is written by hand (no heapq, no sorted()) so the data
structures themselves are the project:

* MinHeap            - array-based binary min-heap (the priority queue)
* PriorityScheduler  - min-heap + hash table (item_id -> node) as in the design doc
* LinkedList         - singly linked list of recently viewed items
* Stack / Queue      - action history (undo) and downlink log
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

# TT&C is always most urgent, then SSTV images, then Codec2/M17 voice/data.
PRIORITY = {"TT&C": 1, "SSTV": 2, "VOICE/DATA": 3}
TYPES = list(PRIORITY)


@dataclass
class Item:
    item_id: str
    type: str
    arrival_time: float          # seconds on the mission clock
    duration_s: float            # airtime needed to downlink the item
    priority: int = 0
    seq: int = 0                 # final tie-breaker -> deterministic order
    index: int = -1              # current position inside the heap array

    def __post_init__(self):
        if self.type not in PRIORITY:
            raise ValueError(f"unknown data type: {self.type!r}")
        self.priority = PRIORITY[self.type]

    @property
    def key(self) -> tuple:
        return (self.priority, self.arrival_time, self.seq)


class MinHeap:
    """Binary min-heap stored in a Python list.

    `on_move(item, idx)` is called whenever an element lands in a slot, which
    lets the owner keep item -> heap-position up to date (needed for O(log n)
    removal of an arbitrary item). `comparisons` counts key comparisons so the
    UI can show the real cost of each operation.
    """

    def __init__(self, on_move: Optional[Callable[[Item, int], None]] = None,
                 data: Optional[list] = None):
        self._a: list[Item] = list(data) if data else []
        self._on_move = on_move
        self.comparisons = 0

    def __len__(self):
        return len(self._a)

    def is_empty(self) -> bool:
        return not self._a

    def peek(self) -> Optional[Item]:
        return self._a[0] if self._a else None

    def items(self) -> list[Item]:
        """Array view of the heap (level order) - not sorted."""
        return list(self._a)

    # -- helpers -----------------------------------------------------------
    def _less(self, i: int, j: int) -> bool:
        self.comparisons += 1
        return self._a[i].key < self._a[j].key

    def _place(self, idx: int, item: Item):
        self._a[idx] = item
        if self._on_move:
            self._on_move(item, idx)

    def _swap(self, i: int, j: int):
        a = self._a
        a[i], a[j] = a[j], a[i]
        if self._on_move:
            self._on_move(a[i], i)
            self._on_move(a[j], j)

    def _sift_up(self, i: int):
        while i > 0:
            parent = (i - 1) // 2
            if self._less(i, parent):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int):
        n = len(self._a)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self._less(left, smallest):
                smallest = left
            if right < n and self._less(right, smallest):
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    # -- public operations -------------------------------------------------
    def push(self, item: Item):                       # O(log n)
        self._a.append(item)
        if self._on_move:
            self._on_move(item, len(self._a) - 1)
        self._sift_up(len(self._a) - 1)

    def pop_min(self) -> Optional[Item]:              # O(log n)
        if not self._a:
            return None
        top = self._a[0]
        last = self._a.pop()
        if self._a:
            self._place(0, last)
            self._sift_down(0)
        if self._on_move:          # a scratch copy must not touch shared nodes
            top.index = -1
        return top

    def remove_at(self, i: int) -> Item:              # O(log n)
        """Remove the element at array position i (used for cancel)."""
        removed = self._a[i]
        last = self._a.pop()
        if i < len(self._a):
            self._place(i, last)
            # the replacement may need to move up OR down (never both)
            self._sift_up(i)
            if self._a[i] is last:
                self._sift_down(i)
        if self._on_move:
            removed.index = -1
        return removed


class PriorityScheduler:
    """Min-heap (transmission order) + hash table (item_id -> node)."""

    def __init__(self):
        self._seq = 0
        self.table: dict[str, Item] = {}              # the hash table
        self.heap = MinHeap(on_move=self._moved)

    @staticmethod
    def _moved(item: Item, idx: int):
        item.index = idx

    def __len__(self):
        return len(self.table)

    def __contains__(self, item_id: str):
        return item_id in self.table

    # insert ----------------------------------------------------------------
    def insert(self, item_id: str, type: str, arrival_time: float,
               duration_s: float, _seq: Optional[int] = None) -> Item:
        if item_id in self.table:
            raise KeyError(f"duplicate item_id: {item_id}")
        if duration_s <= 0:
            raise ValueError("duration must be positive")
        self._seq += 1
        node = Item(item_id, type, arrival_time, duration_s,
                    seq=self._seq if _seq is None else _seq)
        self.heap.push(node)                          # O(log n)
        self.table[item_id] = node                    # O(1)
        return node

    def reinsert(self, node: Item) -> Item:
        """Put a previously removed node back (used by undo / pass skipping)."""
        if node.item_id in self.table:
            raise KeyError(f"duplicate item_id: {node.item_id}")
        node.index = -1
        self.heap.push(node)
        self.table[node.item_id] = node
        return node

    # search ----------------------------------------------------------------
    def search(self, item_id: str) -> Optional[Item]:  # O(1)
        return self.table.get(item_id)

    def position_in_order(self, item_id: str) -> Optional[int]:
        """1-based rank in transmission order (O(n log n); for display only)."""
        for rank, it in enumerate(self.traverse(), 1):
            if it.item_id == item_id:
                return rank
        return None

    # delete ----------------------------------------------------------------
    def delete(self, item_id: str) -> Optional[Item]:  # O(1) + O(log n)
        node = self.table.get(item_id)
        if node is None:
            return None
        self.heap.remove_at(node.index)
        del self.table[item_id]
        return node

    # pop -------------------------------------------------------------------
    def peek(self) -> Optional[Item]:
        return self.heap.peek()

    def get_next_to_transmit(self) -> Optional[Item]:  # O(log n)
        node = self.heap.pop_min()
        if node is not None:
            del self.table[node.item_id]
        return node

    # traverse --------------------------------------------------------------
    def traverse(self) -> list[Item]:                  # O(n log n)
        """All pending items in transmission order, without touching the real heap."""
        tmp = MinHeap(on_move=None, data=self.heap.items())
        out = []
        while not tmp.is_empty():
            out.append(tmp.pop_min())
        return out

    def clone(self) -> "PriorityScheduler":
        other = PriorityScheduler()
        other._seq = self._seq
        for n in self.heap.items():
            other.insert(n.item_id, n.type, n.arrival_time, n.duration_s, _seq=n.seq)
        return other


# ---------------------------------------------------------------------------
# Supporting structures (mirror the Heatwave Watch report: list/stack/queue/set)
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value, nxt=None):
        self.value, self.next = value, nxt


class LinkedList:
    """Singly linked list; newest first. Used for 'recently viewed' items."""

    def __init__(self, capacity: int = 10):
        self.head: Optional[_Node] = None
        self.size = 0
        self.capacity = capacity

    def push_front(self, value):                       # O(1)
        self.head = _Node(value, self.head)
        self.size += 1
        if self.size > self.capacity:                  # drop the oldest, O(k)
            cur = self.head
            for _ in range(self.capacity - 1):
                cur = cur.next
            cur.next = None
            self.size = self.capacity

    def __iter__(self) -> Iterator:
        cur = self.head
        while cur:
            yield cur.value
            cur = cur.next

    def to_list(self) -> list:
        return list(self)


class Stack:
    """LIFO action history; powers 'undo last action'."""

    def __init__(self):
        self._d: list = []

    def push(self, x):
        self._d.append(x)

    def pop(self):
        return self._d.pop() if self._d else None

    def peek(self):
        return self._d[-1] if self._d else None

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)

    def to_list(self) -> list:
        """Top of stack first."""
        return self._d[::-1]


class Queue:
    """FIFO downlink log (what was transmitted, oldest first)."""

    def __init__(self):
        self._d: deque = deque()

    def enqueue(self, x):
        self._d.append(x)

    def dequeue(self):
        return self._d.popleft() if self._d else None

    def pop_newest(self):                              # used only by undo
        return self._d.pop() if self._d else None

    def __len__(self):
        return len(self._d)

    def to_list(self) -> list:
        return list(self._d)
