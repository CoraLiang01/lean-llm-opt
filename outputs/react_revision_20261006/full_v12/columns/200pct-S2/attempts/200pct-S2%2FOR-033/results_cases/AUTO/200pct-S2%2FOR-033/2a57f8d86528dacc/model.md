## Mathematical Model

Sets:
- $T$: set of all 48 half-hour time intervals in the day, indexed by $t$ (from 0 to 47), with time labels and requirements from 44.csv.
- $S$: set of all possible shift start times, $S = T$ (since a shift can start at any interval).

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t \in T$ (from column "Requirement", table_id: file_0_view_0).
- $n = 16$: number of consecutive intervals in an 8-hour shift (since $8$ hours $= 16$ half-hours).

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff whose shift starts at interval $s$.

Objective:
$$
\min \sum_{s \in S} x_s
$$

Constraints:
For every interval $t \in T$,
$$
\sum_{s \in S} x_s \cdot \delta_{t,s} \geq r_t
$$
where
$$
\delta_{t,s} = 
\begin{cases}
1 & \text{if interval } t \text{ is covered by a shift starting at } s \\
0 & \text{otherwise}
\end{cases}
$$

Specifically, a shift starting at $s$ covers intervals $s, s+1, ..., s+15$ (modulo 48). Thus,
$$
\sum_{s \in S: \ t \in \{s, s+1, ..., s+15\} \bmod 48} x_s \geq r_t, \quad \forall t \in T
$$

Variable domains:
$$
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from column "Time" in 44.csv (table_id: file_0_view_0).
- $r_t$: "Requirement" column, 44.csv, table_id: file_0_view_0, row $t$.
- $x_s$: integer variable, number of staff starting at interval $s$.
- Each shift covers 16 consecutive intervals, wrapping around midnight (modulo 48).

All requirements and time intervals are mapped directly from 44.csv (table_id: file_0_view_0).