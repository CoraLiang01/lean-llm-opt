## Mathematical Model

**Sets**
- $T$: set of all time intervals, indexed by $t$ (from the "Time" column in table_id: file_0_view_0, $|T|=48$)
- $S$: set of all possible shift start times, indexed by $s$ (one for each $t \in T$; shifts start at each interval and last 8 consecutive intervals)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from "Requirement" in file_0_view_0)
- $n$: number of intervals in a shift ($n=16$, since each interval is 30 minutes and a shift is 8 hours)
- $|T|$: total number of intervals in a day ($|T|=48$)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff whose shift starts at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For each interval $t \in T$:
$$
\sum_{s \in S: t \in \text{Shift}(s)} x_s \geq r_t
$$
where $\text{Shift}(s) = \{s, s+1, \ldots, s+n-1\}$ (modulo $|T|$), i.e., the set of intervals covered by a shift starting at $s$ (wrap around at midnight).

**Variable Domains**
$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$ and $S$ are both the set of 48 intervals from the "Time" column in table_id: file_0_view_0.
- $r_t$ is the "Requirement" value for each interval $t$ in table_id: file_0_view_0.
- Each shift covers 16 consecutive intervals (8 hours), wrapping around midnight as needed.
- All parameters and indices are mapped directly from file_0_view_0 columns "Time" and "Requirement".