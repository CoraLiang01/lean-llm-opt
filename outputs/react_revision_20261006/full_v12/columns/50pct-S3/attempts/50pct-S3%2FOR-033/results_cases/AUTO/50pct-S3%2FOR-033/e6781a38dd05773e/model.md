## Mathematical Model

Sets:
- $T$: set of 48 half-hour intervals, indexed by $t$ (from 1 to 48), with time labels and requirements from 44.csv.
- $S$: set of possible shift start times, $S = T$ (one possible shift start at each interval).

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t$ (from 44.csv, column "Requirement", table_id: file_0_view_0).
- Each shift covers 16 consecutive intervals (8 hours = 16 half-hours).

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$.

Objective:
$$
\min \sum_{s \in S} x_s
$$

Constraints:
For all $t \in T$,
$$
\sum_{s \in S: \ t \in \{s, s+1, \ldots, s+15\} \pmod{48}} x_s \geq r_t
$$

Variable domains:
$$
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
$$

## Data Mapping

- $T$, $S$: All 48 intervals from 44.csv, column "Time", table_id: file_0_view_0.
- $r_t$: 44.csv, column "Requirement", table_id: file_0_view_0, for each interval $t$.
- Each shift covers 16 consecutive intervals, wrapping around midnight (modulo 48).
- $x_s$: number of waitstaff starting at interval $s$.

This model ensures that at every interval, the sum of waitstaff on duty (those whose 8-hour shift covers that interval) meets or exceeds the required minimum. The objective is to minimize the total number of waitstaff scheduled.