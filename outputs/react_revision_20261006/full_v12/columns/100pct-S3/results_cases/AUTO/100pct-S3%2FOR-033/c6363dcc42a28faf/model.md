## Mathematical Model

Sets:
- $T$: set of time intervals (indexed by $t$), as in 44.csv, $|T|=48$
- $S$: set of possible shift start times (indexed by $s$), $S = T$ (one possible shift start per interval)

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement" in 44.csv)
- Each shift covers 16 consecutive intervals (8 hours × 2 intervals/hour)

Decision variables:
- $x_s \in \mathbb{Z}_+, \ \forall s \in S$: number of waitstaff starting a shift at interval $s$

Objective:
$$
\min \sum_{s \in S} x_s
$$

Constraints:
$$
\sum_{s \in S: t \in \text{Shift}(s)} x_s \geq r_t, \quad \forall t \in T
$$
where $\text{Shift}(s) = \{s, s+1, \ldots, s+15\}$ (modulo 48, i.e., wrap around midnight).

Variable domains:
$$
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
$$

## Data Mapping

- $T$, $S$: All 48 intervals from 44.csv, column "Time", table_id: file_0_view_0
- $r_t$: "Requirement" column, table_id: file_0_view_0, row $t$
- Each $x_s$ corresponds to a shift starting at time interval $s$ ("Time" in 44.csv)
- Each shift covers 16 consecutive intervals, wrapping around the 48-interval day

All data and indices are mapped directly from 44.csv, table_id: file_0_view_0, columns "Time" and "Requirement".