# SatelliteSched – Smart Satellite Data Transmission Schedule
Data Structures project (316U40C302 / 316U50C302) · Saatvik Marathe (16015025034), Dhruvil Mehta (16015025035)

Streamlit app that schedules SomaiyaSat downlink traffic (TT&C > SSTV > Voice/Data) with a
**hand-written min-heap + hash table**, exactly as in the design document.

## Run
```
python -m pip install -r requirements.txt
streamlit run app.py
python -m pytest tests      # data-structure tests (pytest needed)
```
Then click **Load demo scenario** in the sidebar.

## Files
| File | What it holds |
|---|---|
| `app.py` | Streamlit UI (Dashboard, Scheduler, Structure View, Pass Simulator, Benchmark, Activity, About) |
| `ds.py` | MinHeap, PriorityScheduler (heap + dict), LinkedList, Stack, Queue |
| `sim.py` | Pass simulation (heap vs FIFO), demo data, benchmark |
| `tests/test_ds.py` | 12 tests incl. randomised check against a reference model |

## Design-doc operations → code
insert / search / delete / get_next_to_transmit / traverse → `PriorityScheduler` in `ds.py`.
