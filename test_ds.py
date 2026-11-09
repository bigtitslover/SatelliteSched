import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from ds import PriorityScheduler, MinHeap, LinkedList, Stack, Queue, Item
import sim


def build(rows):
    s = PriorityScheduler()
    for r in rows:
        s.insert(*r)
    return s


def heap_ok(s):
    a = s.heap.items()
    for i, x in enumerate(a):
        assert x.index == i, "hash-table node must know its heap slot"
        for c in (2 * i + 1, 2 * i + 2):
            if c < len(a):
                assert x.key <= a[c].key
    assert set(s.table) == {x.item_id for x in a}


def test_ttc_beats_earlier_sstv():
    s = build([("SSTV-1", "SSTV", 0, 100), ("VD-1", "VOICE/DATA", 1, 20), ("TTC-1", "TT&C", 50, 5)])
    assert [i.item_id for i in s.traverse()] == ["TTC-1", "SSTV-1", "VD-1"]
    assert s.get_next_to_transmit().item_id == "TTC-1"


def test_same_priority_is_fifo_then_seq():
    s = build([("A", "SSTV", 5, 10), ("B", "SSTV", 2, 10), ("C", "SSTV", 2, 10)])
    assert [i.item_id for i in s.traverse()] == ["B", "C", "A"]


def test_search_delete_and_duplicates():
    s = build([("A", "TT&C", 0, 5), ("B", "SSTV", 1, 90)])
    assert s.search("B").type == "SSTV" and s.search("Z") is None
    assert s.delete("B").item_id == "B" and s.delete("B") is None
    assert len(s) == 1 and "B" not in s
    with pytest.raises(KeyError):
        s.insert("A", "TT&C", 3, 5)
    with pytest.raises(ValueError):
        s.insert("N", "BOGUS", 3, 5)
    with pytest.raises(ValueError):
        s.insert("N", "TT&C", 3, 0)


def test_empty():
    s = PriorityScheduler()
    assert s.get_next_to_transmit() is None and s.peek() is None and s.traverse() == []


def test_traverse_does_not_disturb_heap():
    s = build(sim.demo_scenario())
    before = [i.item_id for i in s.heap.items()]
    s.traverse()
    assert [i.item_id for i in s.heap.items()] == before
    heap_ok(s)


def test_randomised_against_reference():
    rng = random.Random(42)
    s, ref, n = PriorityScheduler(), {}, 0
    for _ in range(3000):
        op = rng.random()
        if op < 0.5:
            n += 1
            kind = rng.choice(["TT&C", "SSTV", "VOICE/DATA"])
            node = s.insert(f"I{n}", kind, rng.randint(0, 50), 5)
            ref[node.item_id] = node.key
        elif op < 0.75 and ref:
            tid = rng.choice(list(ref))
            assert s.delete(tid) is not None
            del ref[tid]
        elif ref:
            got = s.get_next_to_transmit()
            want = min(ref, key=lambda k: ref[k])
            assert got.item_id == want
            del ref[want]
        if _ % 100 == 0:
            heap_ok(s)
    heap_ok(s)
    assert [i.item_id for i in s.traverse()] == sorted(ref, key=lambda k: ref[k])


def test_remove_middle_needs_sift_up_and_down():
    # removing a node whose replacement is smaller than its new parent
    h = MinHeap()
    for k, pr in enumerate([1, 10, 2, 11, 12, 3, 4]):
        h.push(Item(f"i{k}", ["TT&C", "SSTV", "VOICE/DATA"][pr % 3], pr, 1, seq=k))
    keys = []
    while h.peek():
        keys.append(h.pop_min().key)
    assert keys == sorted(keys)


def test_clone_independent():
    s = build(sim.demo_scenario())
    c = s.clone()
    c.get_next_to_transmit()
    assert len(c) == len(s) - 1
    assert [i.item_id for i in s.traverse()][1:] == [i.item_id for i in c.traverse()]


def test_pass_priority_vs_fifo():
    rows = sim.demo_scenario()
    s = build(rows)
    sent, left = sim.run_priority_pass(s.clone(), 600)
    fifo = sim.run_fifo_pass(list(s.heap.items()), 600)
    ttc = lambda seg: [x for x in seg if x.item.type == "TT&C"]
    assert len(ttc(sent)) == 4                     # every TT&C goes out with the heap
    assert len(ttc(fifo)) < 4                      # FIFO starves some behind SSTV
    assert sum(x.item.duration_s for x in sent) <= 600
    # leftover items are back in the live structure, none lost
    live = s.clone(); sent2, left2 = sim.run_priority_pass(live, 600)
    assert len(live) == len(rows) - len(sent2)
    heap_ok(live)


def test_pass_strict_stops_at_first_misfit():
    s = build([("T", "TT&C", 0, 10), ("S", "SSTV", 1, 100), ("V", "VOICE/DATA", 2, 5)])
    sent, _ = sim.run_priority_pass(s.clone(), 50, fill_gaps=False)
    assert [x.item.item_id for x in sent] == ["T"]
    sent, _ = sim.run_priority_pass(s.clone(), 50, fill_gaps=True)
    assert [x.item.item_id for x in sent] == ["T", "V"]


def test_support_structures():
    ll = LinkedList(capacity=3)
    for x in "abcd":
        ll.push_front(x)
    assert ll.to_list() == ["d", "c", "b"]
    st = Stack(); st.push(1); st.push(2)
    assert st.pop() == 2 and st.peek() == 1 and st.pop() == 1 and st.pop() is None
    q = Queue(); q.enqueue(1); q.enqueue(2)
    assert q.dequeue() == 1 and q.to_list() == [2]


def test_benchmark_runs():
    rows = sim.benchmark(sizes=(200, 400), ops=20)
    assert len(rows) == 2 * 9 and all(r["us"] >= 0 for r in rows)
