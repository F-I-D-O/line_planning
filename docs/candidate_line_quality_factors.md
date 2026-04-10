# Factors affecting candidate line quality (single merged ranking)

One ordered list: **configuration / design** factors and **methodological** differences vs `scripts/random.py` are ranked together by **likely impact** on whether travel requests get a reasonable line. Order is **approximate** (depends on instance symmetry, demand shape, evaluation rules).

---

## 1. Demand flows ignored when generating candidates *(configuration / design)*

**Why it ranks first:** Lines are built from random stops and detour rules, not from **where trips actually occur**. The pool can be “valid” but **misaligned** with real OD patterns.

**Symptom:** Many requests have no reasonable line (coverage, order, or detour).

**Mitigation (conceptual):** Seed stops or corridors from demand; bias or filter lines by high-volume OD or transfer patterns.

---

## 2. Stop pool size: `number_of_stops` in `generate_candidate_lines` *(configuration)*

**What the parameter actually is:** In `lineplanning/candidate_lines.py`, `number_of_stops` is **not** “stops per line” or a per-route cap. It is **how many distinct network (matrix) node IDs** are drawn **uniformly at random without replacement** from the full instance (`0 … node_count−1`) to build the list `stops`. The `CandidateLineGenerator` then treats **only those nodes** as possible stops on candidate lines—endpoints and intermediates for the skeleton are always chosen from this pool. Nothing outside `stops` can appear as a stop on a generated line.

```342:348:lineplanning/candidate_lines.py
    logging.info("Randomly selecting stops")
    potential_stops = [i for i in range(travel_time_provider.get_node_count())]
    stops: list[int] = []
    for _ in range(number_of_stops):
        stop_index = random.randint(0, len(potential_stops) - 1)
        new_stop = potential_stops.pop(stop_index)
        stops.append(new_stop)
```

**Why it ranks high (quality impact):** If `number_of_stops` is **small**, the pool is a **sparse random subset** of the city: most of the network never appears as a line stop, so **coverage** of origins/destinations and useful transfer points is poor no matter how you tune detours or `nb_lines`. If the pool is **larger**, lines can use more of the network—but the set is still a **uniform** sample (see §3), so increasing size helps **coverage** more than **relevance** to demand unless you change how stops are chosen.

**Symptom:** With a small pool, large areas have no nearby line stops; many OD pairs are unserved. With a pool that is large but random, you may cover more ground while still missing **where trips concentrate**.

**Mitigation (conceptual):** Raise `number_of_stops` if the bottleneck is sheer sparsity; combine with **non-uniform** stop choice (§3) if the bottleneck is **wrong** locations.

---

## 3. How stops enter the pool: uniform random draw vs sensible preselection *(configuration)*

**Why it ranks high:** The default mechanism is **uniform sampling** over **all** matrix indices into a pool of size `number_of_stops` (§2). That treats low-activity nodes like major hubs. Lines connect **arbitrary** points instead of **corridors and demand-heavy zones**.

**Symptom:** Geometrically valid lines that are **irrelevant** to most requests.

**Mitigation (conceptual):** Preselect stops (demand, centrality, clustering, terminals)—i.e. change **which** nodes fill the pool, not only **how many**.

---

## 4. Not enough candidate lines (`nb_lines`) *(configuration)*

**Why it matters:** Need **diversity** to cover many OD pairs. Too few lines → many requests still uncovered.

**Symptom:** Coverage improves only slowly as you raise `nb_lines` (especially if §1–§3 are unfixed).

---

## 5. Stops-per-line cap: `min_length` and `max_length` *(configuration)*

These are parameters of **`generate_candidate_lines`** (with module defaults in `candidate_lines.py`) and are stored on **`CandidateLineGenerator`** for each run.

**How this differs from §2:** `number_of_stops` fixes **which nodes may appear** on any line (the pool). `min_length` / `max_length` bound **how many stops a single line may keep** after the skeleton is built—they shape **granularity** of each route (density of stops along the corridor), not **geographic coverage** of the network.

**What the code does:** For each candidate line, a target `length` is drawn uniformly from **`[min_length, max_length]`** (inclusive). The generator concatenates stop sequences from the three skeleton legs, then **repeatedly deletes a random interior stop** until `len(line) ≤ length`. Endpoints are never removed (`index` is in `1 … len(line)−2`).

```287:288:lineplanning/candidate_lines.py
                    length = random.randint(self.min_length, self.max_length)
                    new_line, route = self.generate_new_line_skeleton_manhattan(length)
```

```273:276:lineplanning/candidate_lines.py
        while len(line) > length:
            index = random.randint(1, len(line) - 2)
            line.pop(index)
        return line, route
```

**Subtlety:** The pair is **not** a hard “output must have between min and max stops.” The draw sets only a **ceiling** for shortening. If the skeleton already has **fewer** stops than the drawn `length`, nothing is added—the line can end **shorter than `min_length`**. If the skeleton is **long**, deletion runs until at most `length` stops remain, with **`length` anywhere in `[min_length, max_length]`**, so final stop counts vary line to line.

**Quality impact (different axis from §2–§4):**

- **`max_length` too small:** Lines become **very short** stop sequences; many ODs along the same corridor may not share **intermediate** stops, hurting **ride-along** and transfer-friendly structure even when the pool (§2) is good.
- **`max_length` large / wide range:** Richer lines can carry more stops per route; combined with **random** deletion, you get **variability** (some lines dense, some thinned) rather than a single design choice.
- **`min_length` high:** Only raises the **typical** cap `length` for long skeletons; it does **not** guarantee a minimum number of stops on the output (see above).

**Mitigation (conceptual):** Tune `max_length` (and the range) to match how finely you want lines to sample the corridor; if you need a true minimum number of stops, the current shortening logic would need to enforce it explicitly.

---

## 6. Skeleton validated on the graph vs matrix-only *(methodological — `random.py` vs `candidate_lines.py`)*

This single bucket replaces splitting “start–end rules” and “intermediate graph checks” into separate ranks: it is **one methodological gap** (whether the **road graph** constrains the skeleton).

**`candidate_lines.py`:** Picks `end`, `inter`, `inter_2` using **only** travel times from the matrix (min distance for `end`; detour inequalities for intermediates). **No** checks on the actual shortest paths on `G` when sampling those nodes.

**`scripts/random.py`:** After matrix filters, it **rejects** candidates unless NetworkX shortest paths satisfy:

| Check | Location | What it enforces |
|--------|----------|------------------|
| **Simple direct path** | `generate_new_line`, when accepting `start`/`end` | Shortest path `start → end` has **no repeated nodes** (`len(set(routeA)) == len(routeA)`). That is the **“no self-intersection”** test on the **full** start–end shortest path (a path that revisits a node would fail). |
| **Simple legs + single junction** | Same function, first intermediate | `routeB` = start→inter, `routeC` = inter→end: each leg is simple (`len(set(routeB)) == len(routeB)`, same for `routeC`), and the **union size** condition forces the two paths to overlap on **exactly one** node (the shared endpoint `inter`)—no extra **spurious overlap** between the two segments beyond meeting at `inter`. |
| Same idea | Second intermediate | `routeD` = inter→inter_2, `routeE` = inter_2→end: again simple legs + **exactly one** shared node between the two pieces (`inter_2`). |

**Code references (`scripts/random.py`):**

- Start–end **simple path** (no duplicate nodes on shortest path `start`–`end`):

```22:25:scripts/random.py
        if cost_edges[start,end] >= min_distance and cost_edges[start,end] <= max_travel:
            routeA = nx.shortest_path(bus_network, start, end, weight='length')
            if len(set(routeA)) == len(routeA):
                break
```

- First intermediate: **simple** `routeB` / `routeC` and **controlled overlap** (`unionBC`):

```35:40:scripts/random.py
        if cost_edges[start, inter]  + cost_edges[inter, end] <= cost_edges[start, end] * detour_skeleton:
            routeB = nx.shortest_path(bus_network, start, inter, weight='length')
            routeC = nx.shortest_path(bus_network, inter, end, weight='length')
            unionBC = set(routeB) | set(routeC)
            if len(set(unionBC)) == len(set(routeB)) + len(set(routeC)) - 1 and len(set(routeB)) == len(routeB) and len(set(routeC)) == len(routeC):
                break
```

- Second intermediate: same pattern for `routeD` / `routeE` and `unionDE`:

```50:55:scripts/random.py
        if cost_edges[inter, inter_2]  + cost_edges[inter_2, end] <= cost_edges[inter, end] * detour_skeleton:
            routeD = nx.shortest_path(bus_network, inter, inter_2, weight='length')
            routeE = nx.shortest_path(bus_network, inter_2, end, weight='length')
            unionDE = set(routeD) | set(routeE)
            if len(set(unionDE)) == len(set(routeD)) + len(set(routeE)) - 1 and len(set(routeD)) == len(routeD) and len(set(routeE)) == len(routeE):
                break
```

**Extra in `random.py` only (same item):** Start–end also has a **maximum** matrix cost (`max_travel`), not just a minimum—`candidate_lines.py` has **min only** for the terminal pair.

**Impact:** Matrix-feasible skeletons that are **bad on the network** (long redundant overlap between segments, or a direct shortest path that loops) are **filtered in `random.py`** and **not** in `candidate_lines.py`.

---

## 7. Detour inequalities for intermediates (direction of matrix terms) *(methodological)*

**`candidate_lines.py`:** e.g. `d(start, inter) + d(end, inter)` and `d(inter_2, inter) + d(end, inter_2)`.

**`random.py`:** Forward-trip form `d(start, inter) + d(inter, end)` and `d(inter, inter_2) + d(inter_2, end)`.

**Impact:** Under **asymmetric** travel times, feasible intermediates differ. Under symmetric times, effect is smaller.

---

## 8. Shortest-path edge weight: travel time vs length *(methodological)*

**`candidate_lines.py`:** `travel_time` on `G`. **`random.py`:** `length`.

**Impact:** Time- vs distance-optimal routes **diverge** → different stops on each leg. Quality depends on whether evaluation is time- or distance-centric.

---

## 9. How line length is reduced: random stop deletion vs cost budget *(methodological)*

**`candidate_lines.py`:** Per line, a random cap in **`[min_length, max_length]`** (§5), then **random interior deletion** until under that cap.

**`random.py`:** Prefix truncation using **round-trip cumulative** edge cost vs `max_travel_actual`.

**Impact:** Random deletion can remove **structurally important** stops; budget truncation keeps a **coherent prefix** of the built route. Parameter tuning for stop-count caps is summarized in §5.

---

## 10. One-way stop list vs mirrored two-way line *(methodological)*

**`candidate_lines.py`:** One-way sequence. **`random.py`:** Appends reverse for a two-way route.

**Impact:** If downstream assumes bidirectional service from one sequence, mismatch can hurt opposite-direction or return trips.

---

## 11. No final detour / subtour checks on the assembled line *(methodological)*

**`random.py`:** `detour_coeff` vs chord and duplicate-stop (**subtour**) checks on the final route (with violation flags). **`candidate_lines.py`:** No analogous final filter in the skeleton path that writes `lines.txt`.

**Impact:** Globally poor or repeated-stop lines can still enter the candidate pool.

---

## Summary table (merged ranking)

| Rank | Factor | Kind |
|------|--------|------|
| 1 | Demand-blind generation | Configuration / design |
| 2 | Stop pool size (`number_of_stops`: count of matrix nodes sampled into the stop pool) | Configuration |
| 3 | Uniform random stop draw vs preselection (which nodes fill that pool) | Configuration |
| 4 | Too few candidate lines | Configuration |
| 5 | Stops-per-line cap (`min_length` / `max_length` + random interior deletion) | Configuration |
| 6 | Graph skeleton checks (simple paths, segment overlap, start–end max) vs matrix-only | Methodological |
| 7 | Detour inequality term order (asymmetry) | Methodological |
| 8 | Travel time vs length shortest paths | Methodological |
| 9 | Random interior deletion vs cost-budget truncation | Methodological |
| 10 | One-way vs mirrored two-way | Methodological |
| 11 | Missing final detour / subtour checks | Methodological |

---

*Prioritize demand-aware generation and stop design (§1–§3) before expecting large gains from only increasing `nb_lines`.*
