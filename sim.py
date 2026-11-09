"""Pass simulation, demo data and benchmarks for SatelliteSched."""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

from ds import PRIORITY, Item, PriorityScheduler

# Typical airtimes in seconds (simulated, not measured from a real satellite).
DURATION_RANGE = {"TT&C": (4, 15), "SSTV": (90, 130), "VOICE/DATA": (10, 40)}


@dataclass
class Segment:
    item: Item
    start: float
    end: float

    @property
    def latency(self) -> float:
        """Wait between the item being generated and its transmission start."""
        return self.start - self.item.arrival_time


def run_priority_pass(sched: PriorityScheduler, window_s: float, pass_start: float = 0.0,
                      fill_gaps: bool = True) -> tuple[list[Segment], list[Item]]:
    """Transmit with the min-heap. MUTATES `sched` (pass a clone for a preview).

    Items that do not fit in the time left are re-inserted afterwards; with
    fill_gaps=False the pass stops at the first item that does not fit.
    """
    t, sent, skipped = 0.0, [], []
    while True:
        node = sched.get_next_to_transmit()            # O(log n)
        if node is None:
            break
        if t + node.duration_s <= window_s:
            sent.append(Segment(node, pass_start + t, pass_start + t + node.duration_s))
            t += node.duration_s
        elif fill_gaps:
            skipped.append(node)
        else:
            skipped.append(node)
            break
    for n in skipped:
        sched.reinsert(n)
    return sent, [n for n in skipped]


def run_fifo_pass(items: list[Item], window_s: float, pass_start: float = 0.0,
                  fill_gaps: bool = True) -> list[Segment]:
    """Baseline: send in arrival order, ignoring priority."""
    q = deque(sorted(items, key=lambda i: (i.arrival_time, i.seq)))
    t, sent = 0.0, []
    while q:
        node = q.popleft()
        if t + node.duration_s <= window_s:
            sent.append(Segment(node, pass_start + t, pass_start + t + node.duration_s))
            t += node.duration_s
        elif not fill_gaps:
            break
    return sent


def demo_scenario(seed: int = 7) -> list[tuple[str, str, float, float]]:
    """A scenario that shows the problem: big SSTV images arrive first, urgent
    TT&C arrives later. Returns (item_id, type, arrival_time, duration_s)."""
    rng = random.Random(seed)
    rows, n = [], {"TT&C": 0, "SSTV": 0, "VOICE/DATA": 0}
    prefix = {"TT&C": "TTC", "SSTV": "SSTV", "VOICE/DATA": "VD"}

    def add(kind, t):
        n[kind] += 1
        lo, hi = DURATION_RANGE[kind]
        rows.append((f"{prefix[kind]}-{n[kind]:03d}", kind, float(t), float(rng.randint(lo, hi))))

    for t in (0, 20, 45, 70, 95):
        add("SSTV", t)
    for t in (30, 60, 100, 130, 160, 190):
        add("VOICE/DATA", t)
    for t in (50, 110, 170, 230):
        add("TT&C", t)
    rows.sort(key=lambda r: r[2])
    return rows


def random_item(kind: str, t: float, counters: dict, rng: random.Random):
    counters[kind] = counters.get(kind, 0) + 1
    prefix = {"TT&C": "TTC", "SSTV": "SSTV", "VOICE/DATA": "VD"}[kind]
    lo, hi = DURATION_RANGE[kind]
    return f"{prefix}-{counters[kind]:03d}", kind, float(t), float(rng.randint(lo, hi))


# ---------------------------------------------------------------------------
# Benchmark: chosen design vs the two rejected options from the design doc
# ---------------------------------------------------------------------------
def _make_items(n: int, rng: random.Random) -> list[Item]:
    kinds = list(PRIORITY)
    return [Item(f"X{i}", rng.choice(kinds), float(i), 10.0, seq=i) for i in range(n)]


def benchmark(sizes=(500, 1000, 2000, 4000, 8000), ops: int = 100, seed: int = 1):
    """Average microseconds per operation for each design. Returns list of dict rows."""
    rng = random.Random(seed)
    rows = []
    for n in sizes:
        items = _make_items(n, rng)
        ids = [it.item_id for it in items]
        targets = rng.sample(ids, min(ops, n))

        # --- Option 3 (chosen): heap + hash --------------------------------
        s = PriorityScheduler()
        t0 = time.perf_counter()
        for it in items:
            s.insert(it.item_id, it.type, it.arrival_time, it.duration_s)
        ins = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for tid in targets:
            s.delete(tid)
        cancel = (time.perf_counter() - t0) / len(targets)
        k = min(ops, len(s))
        t0 = time.perf_counter()
        for _ in range(k):
            s.get_next_to_transmit()
        pop = (time.perf_counter() - t0) / k
        rows += [dict(n=n, design="Min-Heap + Hash", op="Insert", us=ins * 1e6),
                 dict(n=n, design="Min-Heap + Hash", op="Pop next", us=pop * 1e6),
                 dict(n=n, design="Min-Heap + Hash", op="Cancel", us=cancel * 1e6)]

        # --- Option 1: FIFO queue (fast but ignores priority) ----------------
        q = deque()
        t0 = time.perf_counter()
        for it in items:
            q.append(it)
        ins = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for tid in targets:
            for x in q:                                 # O(n) scan
                if x.item_id == tid:
                    q.remove(x)
                    break
        cancel = (time.perf_counter() - t0) / len(targets)
        k = min(ops, len(q))
        t0 = time.perf_counter()
        for _ in range(k):
            q.popleft()
        pop = (time.perf_counter() - t0) / k
        rows += [dict(n=n, design="FIFO queue", op="Insert", us=ins * 1e6),
                 dict(n=n, design="FIFO queue", op="Pop next", us=pop * 1e6),
                 dict(n=n, design="FIFO queue", op="Cancel", us=cancel * 1e6)]

        # --- Option 2: array re-sorted every time ----------------------------
        arr = []
        t0 = time.perf_counter()
        for it in items:
            arr.append(it)
        ins = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for tid in targets:
            for i, x in enumerate(arr):                 # O(n) scan
                if x.item_id == tid:
                    del arr[i]
                    break
        cancel = (time.perf_counter() - t0) / len(targets)
        k = min(20, len(arr))                           # each pop re-sorts: keep it short
        t0 = time.perf_counter()
        for _ in range(k):
            arr.sort(key=lambda x: x.key)               # O(n log n)
            arr.pop(0)
        pop = (time.perf_counter() - t0) / k
        rows += [dict(n=n, design="Re-sorted array", op="Insert", us=ins * 1e6),
                 dict(n=n, design="Re-sorted array", op="Pop next", us=pop * 1e6),
                 dict(n=n, design="Re-sorted array", op="Cancel", us=cancel * 1e6)]
    return rows
