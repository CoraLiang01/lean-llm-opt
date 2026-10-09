## Mathematical Model

**Sets**
- $T$: set of time intervals, indexed by $t$ (from 0 to 47, each representing a 30-minute interval; see Data Mapping for exact labels)
- $S$: set of possible shift start times, indexed by $s$ (also 0 to 47, one for each interval)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from Requirement column, table_id: file_0_view_0)
- $n$: number of intervals in a shift ($n = 16$, since 8 hours = 16 half-hour intervals)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff whose shift starts at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For each interval $t \in T$:
$$
\sum_{s \in S} a_{s,t} x_s \geq r_t
$$
where
$$
a_{s,t} = 
\begin{cases}
1 & \text{if interval } t \text{ is covered by a shift starting at } s \\
0 & \text{otherwise}
\end{cases}
$$
Specifically, $a_{s,t} = 1$ if $t \in \{s, (s+1) \bmod 48, \ldots, (s+15) \bmod 48\}$.

**Variable domains**
$$
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from the "Time" column in table_id: file_0_view_0, i.e., $T = S = \{0,1,\ldots,47\}$, with mapping $t \leftrightarrow$ "Time" label.
- $r_t$: "Requirement" column, table_id: file_0_view_0, row $t$.
- $n = 16$ (since each shift is 8 hours, and each interval is 0.5 hours).
- $a_{s,t}$: 1 if $t \in \{s, (s+1) \bmod 48, \ldots, (s+15) \bmod 48\}$, 0 otherwise.
- $x_s$: number of waitstaff starting at interval $s$.

**All data is from table_id: file_0_view_0, columns "Time" and "Requirement".**