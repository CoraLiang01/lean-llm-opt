## Mathematical Model

Sets:
- $T$: set of all 48 half-hour time intervals, indexed by $t$ (from 0 to 47), with labels and requirements from 44.csv.
- $S$: set of possible shift start times, $S = T$ (since a shift can start at any interval).

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement", table_id: file_0_view_0).
- Each shift covers 16 consecutive intervals (8 hours $\times$ 2 per hour).

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$.

Objective:
$$
\min \sum_{s \in S} x_s
$$

Constraints:
For every interval $t \in T$,
$$
\sum_{s \in S: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t
$$

Variable domains:
$$
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from 44.csv, column "Time", table_id: file_0_view_0.
- $r_t$: "Requirement" column, table_id: file_0_view_0, for each $t$.
- Each shift covers 16 consecutive intervals, wrapping around midnight (modulo 48).
- $x_s$: integer variable, number of waitstaff starting at interval $s$.

---

Minimize the total number of waitstaff, ensuring at every interval the sum of all staff on duty (those whose 8-hour shift covers that interval) meets or exceeds the required minimum.