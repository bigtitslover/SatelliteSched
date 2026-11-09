"""SatelliteSched - Smart Satellite Data Transmission Schedule.

Streamlit front-end. All data-structure logic lives in ds.py / sim.py.
Run with:  streamlit run app.py
"""
import math
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sim
from ds import PRIORITY, TYPES, LinkedList, PriorityScheduler, Queue, Stack

st.set_page_config(page_title="SatelliteSched", page_icon="🛰️", layout="wide")

COLORS = {"TT&C": "#ff6b57", "SSTV": "#4da3ff", "VOICE/DATA": "#7bd88f"}
LABEL = {"TT&C": "TT&C (telemetry / health)", "SSTV": "SSTV image", "VOICE/DATA": "Codec2 / M17 voice-data"}
TYPICAL = {k: round(sum(v) / 2) for k, v in sim.DURATION_RANGE.items()}
PAGES = ["Dashboard", "Scheduler", "Structure View", "Pass Simulator", "Benchmark", "Activity", "About"]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if "sched" not in ss:
        ss.sched = PriorityScheduler()
        ss.undo = Stack()            # state-changing actions (for undo)
        ss.downlink = Queue()        # transmitted items, oldest first
        ss.recent = LinkedList(10)   # recently searched items
        ss.clock = 0.0
        ss.counters = {}
        ss.bench = None


def next_id(kind):
    prefix = {"TT&C": "TTC", "SSTV": "SSTV", "VOICE/DATA": "VD"}[kind]
    c = st.session_state.counters
    while True:
        c[kind] = c.get(kind, 0) + 1
        cand = f"{prefix}-{c[kind]:03d}"
        if cand not in st.session_state.sched:
            return cand


def add_item(item_id, kind, arrival, duration):
    ss = st.session_state
    before = ss.sched.heap.comparisons
    node = ss.sched.insert(item_id, kind, float(arrival), float(duration))
    ss.undo.push(("INSERT", node.item_id))
    ss.clock = max(ss.clock, float(arrival))
    return node, ss.sched.heap.comparisons - before


def undo_last():
    ss = st.session_state
    act = ss.undo.pop()
    if act is None:
        return "Nothing to undo."
    op, payload = act
    if op == "INSERT":
        ss.sched.delete(payload)
        return f"Undid insert of {payload}."
    ss.sched.reinsert(payload)
    if op == "TRANSMIT":
        ss.downlink.pop_newest()
    return f"Undid {op.lower()} of {payload.item_id}."


def load_demo():
    ss = st.session_state
    ss.sched, ss.undo, ss.downlink, ss.recent = PriorityScheduler(), Stack(), Queue(), LinkedList(10)
    ss.counters = {}
    for r in sim.demo_scenario():
        ss.sched.insert(*r)
    ss.clock = max(n.arrival_time for n in ss.sched.heap.items())


def clear_all():
    ss = st.session_state
    ss.sched, ss.undo, ss.downlink, ss.recent = PriorityScheduler(), Stack(), Queue(), LinkedList(10)
    ss.clock, ss.counters = 0.0, {}


# --------------------------------------------------------------------------
# Small view helpers
# --------------------------------------------------------------------------
COLS = ["Rank", "Item ID", "Type", "Priority", "Arrival (s)", "Airtime (s)", "Heap slot"]


def to_df(items):
    return pd.DataFrame(
        [[i + 1, x.item_id, x.type, x.priority, x.arrival_time, x.duration_s, x.index]
         for i, x in enumerate(items)], columns=COLS)


def show_table(items):
    st.dataframe(to_df(items), hide_index=True)


def heap_figure(sched, max_nodes=31):
    arr = sched.heap.items()[:max_nodes]
    pos = {}
    for i in range(len(arr)):
        lvl = (i + 1).bit_length() - 1
        w = 2 ** lvl
        pos[i] = ((i + 1 - w + 0.5) / w, -lvl)
    ex, ey = [], []
    for i in range(1, len(arr)):
        p = (i - 1) // 2
        ex += [pos[p][0], pos[i][0], None]
        ey += [pos[p][1], pos[i][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#6b7280", width=1),
                               hoverinfo="skip", showlegend=False))
    for kind in TYPES:
        idx = [i for i, n in enumerate(arr) if n.type == kind]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in idx], y=[pos[i][1] for i in idx], mode="markers+text",
            text=[arr[i].item_id for i in idx], textposition="bottom center", textfont=dict(size=10),
            marker=dict(size=22, color=COLORS[kind], line=dict(color="#0e1117", width=1)),
            name=kind,
            hovertext=[f"slot {i} | {arr[i].item_id}<br>key (priority, arrival, seq) = {arr[i].key}"
                       for i in idx], hoverinfo="text"))
    fig.update_xaxes(visible=False, range=[-0.02, 1.02])
    fig.update_yaxes(visible=False)
    fig.update_layout(height=120 + 110 * max(1, math.ceil(math.log2(len(arr) + 1))) if arr else 200,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
    return fig


def gantt_figure(runs, window, start):
    fig = go.Figure()
    for kind in TYPES:
        ys, xs, bases, texts, hov = [], [], [], [], []
        for policy, segs in runs.items():
            for sg in segs:
                if sg.item.type == kind:
                    ys.append(policy)
                    xs.append(sg.end - sg.start)
                    bases.append(sg.start)
                    texts.append(sg.item.item_id)
                    hov.append(f"{sg.item.item_id} ({kind})<br>sent {sg.start:.0f}s-{sg.end:.0f}s"
                               f"<br>waited {sg.latency:.0f}s since generated")
        fig.add_trace(go.Bar(orientation="h", y=ys, x=xs, base=bases, name=kind, text=texts,
                             textposition="inside", insidetextanchor="middle", textfont=dict(size=10),
                             marker=dict(color=COLORS[kind], line=dict(color="#0e1117", width=1)),
                             hovertext=hov, hoverinfo="text"))
    fig.update_layout(barmode="overlay", height=280, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(range=[start, start + window], title="Mission time (s)"),
                      legend=dict(orientation="h", y=1.15))
    return fig


def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.title("🛰️ SatelliteSched")
        st.caption("Smart Satellite Data Transmission Schedule")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.subheader("Queue")
        counts = st.empty()      # filled at the end of the run so it is never stale
        if st.button("Load demo scenario", help="5 SSTV + 6 voice/data arrive before 4 TT&C items"):
            load_demo()
            st.rerun()
        if st.button("Clear everything"):
            clear_all()
            st.rerun()
        with st.expander("Import / export CSV"):
            st.download_button("Download pending (CSV)",
                               to_df(ss.sched.traverse()).to_csv(index=False), "pending_items.csv", "text/csv")
            up = st.file_uploader("CSV with: item_id, type, arrival_time, duration_s", type="csv")
            if up is not None and st.button("Import file"):
                import_csv(up)
    return page, counts


def import_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read CSV: {e}")
        return
    need = {"item_id", "type", "arrival_time", "duration_s"}
    if not need.issubset(df.columns):
        st.error(f"Missing column(s): {', '.join(sorted(need - set(df.columns)))}")
        return
    ok, bad = 0, []
    for r in df.itertuples():
        try:
            add_item(str(r.item_id), str(r.type), float(r.arrival_time), float(r.duration_s))
            ok += 1
        except (KeyError, ValueError) as e:
            bad.append(f"{r.item_id}: {e}")
    st.success(f"Imported {ok} item(s).")
    if bad:
        st.warning("Skipped " + str(len(bad)) + " row(s):\n\n" + "\n".join(bad[:10]))


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_dashboard():
    ss = st.session_state
    s = ss.sched
    st.title("Mission Dashboard")
    st.write("SomaiyaSat must downlink **TT&C → SSTV → Voice/Data**, in that order of urgency, "
             "during a short ground-station pass.")
    if len(s) == 0:
        st.info("The queue is empty. Load the demo scenario or add items on the Scheduler page.")
        if st.button("Load demo scenario", key="demo_dash"):
            load_demo()
            st.rerun()
        return
    items = s.heap.items()
    counts = {k: sum(1 for x in items if x.type == k) for k in TYPES}
    air = {k: sum(x.duration_s for x in items if x.type == k) for k in TYPES}
    nxt = s.peek()
    c = st.columns(5)
    c[0].metric("Pending items", len(s))
    c[1].metric("TT&C", counts["TT&C"])
    c[2].metric("SSTV", counts["SSTV"])
    c[3].metric("Voice / Data", counts["VOICE/DATA"])
    c[4].metric("Pending airtime", f"{sum(air.values()):.0f} s")
    st.success(f"Next to transmit: **{nxt.item_id}** ({nxt.type}, {nxt.duration_s:.0f} s) - "
               f"found with a single peek of the heap root, O(1).")
    left, right = st.columns(2)
    df = pd.DataFrame({"Type": TYPES, "Items": [counts[k] for k in TYPES],
                       "Airtime (s)": [air[k] for k in TYPES]})
    with left:
        f = px.bar(df, x="Type", y="Items", color="Type", color_discrete_map=COLORS, title="Pending items by type")
        f.update_layout(showlegend=False, height=300)
        st.plotly_chart(f)
    with right:
        f = px.bar(df, x="Type", y="Airtime (s)", color="Type", color_discrete_map=COLORS,
                   title="Pending airtime by type")
        f.update_layout(showlegend=False, height=300)
        st.plotly_chart(f)
    st.subheader("Next 8 in transmission order")
    show_table(s.traverse()[:8])


def page_scheduler():
    ss = st.session_state
    s = ss.sched
    st.title("Scheduler")
    st.caption("Each tab is one core operation from the design document. The cost line shows real "
               "key comparisons performed by the heap.")
    t_ins, t_search, t_cancel, t_next, t_all = st.tabs(
        ["Insert", "Search", "Cancel", "Transmit next", "Traverse"])

    with t_ins:
        kind = st.selectbox("Data type", TYPES, format_func=lambda k: f"{LABEL[k]} - priority {PRIORITY[k]}")
        c1, c2, c3 = st.columns(3)
        item_id = c1.text_input("Item ID (blank = auto)", "").strip()
        arrival = c2.number_input("Arrival time (s)", min_value=0.0, value=float(ss.clock), step=5.0)
        dur = c3.number_input("Airtime needed (s)", min_value=1.0, value=float(TYPICAL[kind]), step=1.0)
        if st.button("Insert item", type="primary"):
            iid = item_id or next_id(kind)
            try:
                node, cost = add_item(iid, kind, arrival, dur)
                st.success(f"Inserted **{iid}** (priority {node.priority}) at heap slot {node.index}. "
                           f"Heap: {cost} comparison(s) → O(log n). Hash table: 1 write → O(1).")
            except (KeyError, ValueError) as e:
                st.error(str(e).strip("'"))

    with t_search:
        q = st.text_input("Item ID to look up", key="q_search").strip()
        if st.button("Search"):
            node = s.search(q)                                  # O(1)
            if node is None:
                st.warning(f"{q or '(empty)'} is NOT_FOUND (hash lookup, O(1)).")
            else:
                ss.recent.push_front(node.item_id)
                rank = s.position_in_order(q)
                st.success(f"**{node.item_id}** is pending - {node.type}, priority {node.priority}, "
                           f"arrived {node.arrival_time:.0f}s, needs {node.duration_s:.0f}s airtime. "
                           f"Heap slot {node.index}. Position in transmission order: {rank} of {len(s)}.")

    with t_cancel:
        if len(s) == 0:
            st.info("Nothing pending.")
        else:
            ids = sorted(s.table)
            target = st.selectbox("Pending item to cancel", ids)
            if st.button("Cancel item"):
                before = s.heap.comparisons
                node = s.delete(target)
                ss.undo.push(("CANCEL", node))
                st.success(f"Cancelled **{target}**: located via hash table in O(1), removed from the heap in "
                           f"O(log n) ({s.heap.comparisons - before} comparison(s)).")

    with t_next:
        nxt = s.peek()
        st.write("Next up: " + (f"**{nxt.item_id}** ({nxt.type})" if nxt else "nothing"))
        if st.button("Pop & transmit next item", type="primary"):
            before = s.heap.comparisons
            node = s.get_next_to_transmit()
            if node is None:
                st.warning("Queue is empty - returned NONE.")
            else:
                ss.downlink.enqueue(node)
                ss.undo.push(("TRANSMIT", node))
                st.success(f"Transmitted **{node.item_id}** ({node.type}). "
                           f"{s.heap.comparisons - before} comparison(s) → O(log n).")

    with t_all:
        st.write("Full pending list in transmission order (copy of the heap is drained; the real heap is untouched).")
        show_table(s.traverse())


def page_structure():
    s = st.session_state.sched
    st.title("Structure View")
    st.write("See the two structures behind the scheduler. The heap decides **what goes next**; the hash table "
             "finds **any item instantly**, and stores where it sits in the heap.")
    if len(s) == 0:
        st.info("Queue is empty - load the demo scenario from the sidebar.")
        return
    st.subheader("Min-heap (binary tree view)")
    st.plotly_chart(heap_figure(s))
    if len(s) > 31:
        st.caption("Showing the top 31 nodes (5 levels).")
    st.caption("Smaller key = higher in the tree. Key = (priority, arrival time, sequence).")
    a, b = st.columns(2)
    with a:
        st.subheader("Heap array")
        st.dataframe(pd.DataFrame([[i, x.item_id, str(x.key)] for i, x in enumerate(s.heap.items())],
                                  columns=["Slot", "Item ID", "Key"]), hide_index=True)
    with b:
        st.subheader("Hash table")
        st.dataframe(pd.DataFrame([[k, v.index] for k, v in s.table.items()],
                                  columns=["item_id (key)", "Heap slot (value)"]), hide_index=True)


def page_pass():
    ss = st.session_state
    s = ss.sched
    st.title("Pass Simulator")
    st.write("A ground-station pass is short. Compare what actually gets downlinked with the **min-heap** "
             "against a plain **FIFO** queue.")
    if len(s) == 0:
        st.info("Queue is empty - load the demo scenario from the sidebar.")
        return
    last = max(n.arrival_time for n in s.heap.items())
    c1, c2, c3 = st.columns(3)
    window = c1.number_input("Pass length (s)", min_value=30.0, value=600.0, step=30.0)
    start = c2.number_input("Pass starts at (s)", min_value=float(last), value=float(last), step=10.0,
                            help="Cannot be earlier than the last item's arrival.")
    fill = c3.checkbox("Fill gaps with smaller items", value=True,
                       help="If the next item does not fit in the time left, try lower-priority items that do.")
    items = s.heap.items()
    pri_segs, _ = sim.run_priority_pass(s.clone(), window, start, fill)
    fifo_segs = sim.run_fifo_pass(items, window, start, fill)

    total_ttc = sum(1 for x in items if x.type == "TT&C")

    def summary(segs):
        t = [x for x in segs if x.item.type == "TT&C"]
        return (f"{len(t)} of {total_ttc}", f"{min((x.latency for x in t), default=float('nan')):.0f} s" if t else "-",
                f"{sum(x.end - x.start for x in segs):.0f} s", f"{len(items) - len(segs)}")

    st.plotly_chart(gantt_figure({"Min-Heap (priority)": pri_segs, "FIFO (arrival order)": fifo_segs}, window, start))
    m1, m2 = st.columns(2)
    for col, name, segs in ((m1, "Min-Heap (priority)", pri_segs), (m2, "FIFO (arrival order)", fifo_segs)):
        a, b, c, d = summary(segs)
        with col:
            st.subheader(name)
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("TT&C sent", a)
            x2.metric("1st TT&C wait", b)
            x3.metric("Airtime used", c)
            x4.metric("Left over", d)
    st.divider()
    if st.button("Execute this pass on the live queue", type="primary"):
        sent, _ = sim.run_priority_pass(s, window, start, fill)
        for sg in sent:
            ss.downlink.enqueue(sg.item)
            ss.undo.push(("TRANSMIT", sg.item))
        st.success(f"Transmitted {len(sent)} item(s); {len(s)} still pending. See the Activity page.")
        st.caption("Press 'Undo' on the Activity page to put items back one at a time.")


def page_benchmark():
    ss = st.session_state
    st.title("Benchmark")
    st.write("Measures the three designs compared in the design document. Average cost per operation, "
             "in microseconds, on this machine.")
    sizes = st.multiselect("Queue sizes (n)", [500, 1000, 2000, 4000, 8000, 16000], default=[500, 2000, 8000])
    if st.button("Run benchmark", type="primary") and sizes:
        with st.spinner("Timing..."):
            ss.bench = pd.DataFrame(sim.benchmark(tuple(sorted(sizes))))
    if ss.bench is None:
        st.info("Press **Run benchmark**.")
        return
    f = px.line(ss.bench, x="n", y="us", color="design", facet_col="op", markers=True, log_y=True,
                labels={"us": "µs per operation (log)", "n": "items in queue"},
                category_orders={"op": ["Insert", "Pop next", "Cancel"]})
    f.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(f)
    st.dataframe(ss.bench.round(2), hide_index=True)
    st.caption("FIFO pops fastest but sends in the wrong order, and cancelling needs an O(n) scan. Re-sorting an "
               "array makes every pop O(n log n). Only Min-Heap + Hash keeps insert, pop and cancel all cheap "
               "while respecting priority.")


def page_activity():
    ss = st.session_state
    st.title("Activity")
    a, b = st.columns(2)
    with a:
        st.subheader("Action history (stack, LIFO)")
        rows = [[op, (p if isinstance(p, str) else p.item_id)] for op, p in ss.undo.to_list()]
        st.dataframe(pd.DataFrame(rows, columns=["Action", "Item"]), hide_index=True)
        b1, b2 = st.columns(2)
        if b1.button("Undo last action"):
            st.info(undo_last())
        if b2.button("Clear history"):
            ss.undo.clear()
            st.info("History cleared.")
    with b:
        st.subheader("Downlink log (queue, FIFO)")
        show_log = ss.downlink.to_list()
        st.dataframe(pd.DataFrame([[i + 1, x.item_id, x.type, x.duration_s] for i, x in enumerate(show_log)],
                                  columns=["#", "Item ID", "Type", "Airtime (s)"]), hide_index=True)
    c, d = st.columns(2)
    with c:
        st.subheader("Recently searched (linked list)")
        r = ss.recent.to_list()
        st.write(" → ".join(r) if r else "Nothing searched yet.")
    with d:
        st.subheader("Data types pending (set)")
        kinds = {x.type for x in ss.sched.heap.items()}
        st.write(", ".join(sorted(kinds, key=lambda k: PRIORITY[k])) if kinds else "None.")


def page_about():
    st.title("About the Project")
    st.markdown("""
**SatelliteSched - Smart Satellite Data Transmission Schedule**
Data Structures (316U40C302 / 316U50C302) · Case study ref. KJS-SRS-01
Saatvik Marathe (16015025034) · Dhruvil Mehta (16015025035)

**Problem.** SomaiyaSat, a PocketQube, has little power and memory and only short passes over the ground station.
It must send TT&C (health), SSTV images and Codec2/M17 voice/data. A FIFO queue lets a long SSTV image block urgent
TT&C. The scheduler must always send the most important item first, pick the next item quickly, and let any pending
item be checked or cancelled without slowing everything down.
""")
    st.subheader("Why Min-Heap + Hash Table")
    st.dataframe(pd.DataFrame([
        ["Pick next item", "O(1) but wrong order", "O(n log n) re-sort", "O(log n)"],
        ["Add item", "O(1)", "O(1), list unsorted again", "O(log n)"],
        ["Cancel / find item", "O(n) scan", "O(n) scan", "O(1) lookup + O(log n) heap removal"],
        ["Extra memory", "Least", "Least", "A bit more (heap + table)"],
    ], columns=["Requirement", "FIFO queue", "Re-sorted array", "Min-Heap + Hash Table (chosen)"]),
        hide_index=True)
    st.subheader("Data structures in the application")
    st.dataframe(pd.DataFrame([
        ["Min-Heap (priority queue)", "Transmission order; key = (priority, arrival, seq)", "push / pop O(log n), peek O(1)"],
        ["Hash table (dict)", "item_id → node (which knows its heap slot)", "search / lookup O(1) average"],
        ["Stack", "Action history, powers Undo", "push / pop O(1)"],
        ["Queue", "Downlink log of transmitted items", "enqueue / dequeue O(1)"],
        ["Singly linked list", "Recently searched items (max 10)", "insert at front O(1)"],
        ["Set", "Distinct data types currently pending", "membership O(1) average"],
    ], columns=["Structure", "Role", "Complexity"]), hide_index=True)
    st.subheader("Priorities")
    st.write("TT&C = 1 (most urgent), SSTV = 2, Codec2/M17 voice/data = 3. Ties are broken by earlier arrival, "
             "then by insertion order.")
    st.caption("Airtimes and arrivals here are simulated for demonstration, not measured from real satellite hardware. "
               "The heap and linked list are implemented by hand in ds.py; the hash table is Python's dict.")


def main():
    init_state()
    page, counts = sidebar()
    {"Dashboard": page_dashboard, "Scheduler": page_scheduler, "Structure View": page_structure,
     "Pass Simulator": page_pass, "Benchmark": page_benchmark, "Activity": page_activity,
     "About": page_about}[page]()
    with counts.container():
        c1, c2 = st.columns(2)
        c1.metric("Pending", len(st.session_state.sched))
        c2.metric("Sent", len(st.session_state.downlink))


main()
