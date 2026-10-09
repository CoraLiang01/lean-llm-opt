## Mathematical Model

Sets:
- $S$: set of all shift start times (indexed by $s$), $|S| = 48$ (one per 30-minute interval)
- $T$: set of all time intervals (indexed by $t$), $|T| = 48$ (one per 30-minute interval)

Parameters (from 44.csv, table_id: file_0_view_0):
- $r_t$: minimum number of waitstaff required in interval $t$ (column "Requirement", row $t$)
- Each shift $s$ covers 16 consecutive intervals: $C(s) = \{s, s+1, ..., s+15\}$ (modulo 48, i.e., wrap around midnight)

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

Objective:
$$
\min \sum_{s \in S} x_s
$$

Subject to:
$$
\sum_{s: t \in C(s)} x_s \geq r_t, \quad \forall t \in T
$$

$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

## Data Mapping

- $S$, $T$: All 48 intervals from "Time" in 44.csv (table_id: file_0_view_0, column "Time")
- $r_t$: "Requirement" in 44.csv (table_id: file_0_view_0, column "Requirement", row $t$)
- Each $x_s$ corresponds to a shift starting at time interval $s$ (table_id: file_0_view_0, column "Time", row $s$)
- Each shift covers 8 hours = 16 consecutive 30-minute intervals, wrapping around midnight as needed.