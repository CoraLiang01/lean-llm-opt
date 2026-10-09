## Mathematical Model

Sets:
- $T$: set of time intervals (indexed by $t$), as in 44.csv, $|T|=48$
- $S$: set of possible shift start times (indexed by $s$), $S = T$

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t$ (from 44.csv, column "Requirement", table_id: file_0_view_0)
- Each shift covers 8 consecutive intervals (since each interval is 30 minutes, 8 hours = 16 intervals)

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

Objective:
$$
\min \sum_{s \in S} x_s
$$

Constraints:
For each interval $t \in T$:
$$
\sum_{s \in S: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t
$$

Variable domains:
$$
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from 44.csv, column "Time", table_id: file_0_view_0
- $r_t$: 44.csv, column "Requirement", table_id: file_0_view_0, row $t$
- Each $x_s$ corresponds to the number of waitstaff starting at time interval $s$ (from 44.csv, column "Time", table_id: file_0_view_0)
- Each shift covers 16 consecutive intervals (8 hours), wrapping around midnight as needed

All data and indices are mapped directly from 44.csv (table_id: file_0_view_0).