## Mathematical Model

**Sets**
- $T$: set of all 48 half-hour time intervals in the day, indexed by $t$ (from the "Time" column in 44.csv, table_id: file_0_view_0)
- $S$: set of all 48 possible shift start times, indexed by $s$ (one for each $t$; a shift can start at any interval)

**Parameters**
- $r_t$: minimum number of waitstaff required during interval $t$ (from "Requirement" column, table_id: file_0_view_0)
- $n = 48$: total number of intervals in the day
- $L = 16$: number of consecutive intervals in an 8-hour shift (since each interval is 0.5 hours, $8/0.5 = 16$)

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
A shift starting at $s$ covers intervals $s, s+1, \ldots, s+L-1$ (modulo $n$ to wrap around midnight).

**Variable Domains**
$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from "Time" in 44.csv (table_id: file_0_view_0)
- $r_t$: "Requirement" column, table_id: file_0_view_0, row $t$
- $a_{s,t}$: $1$ if $t$ is within the 16 consecutive intervals starting at $s$ (modulo 48), $0$ otherwise
- $x_s$: number of waitstaff starting at interval $s$ (decision variable)