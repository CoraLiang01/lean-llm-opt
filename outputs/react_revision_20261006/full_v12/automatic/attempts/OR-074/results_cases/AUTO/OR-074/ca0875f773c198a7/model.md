## Mathematical Model

Sets:
- $T = \{0,1,\ldots,47\}$: index of 30-minute intervals (from 2:00am–2:30am, ..., 1:30am–2:00am), as in 44.csv.

Parameters (from 44.csv, table_id: file_0_view_0):
- $r_t$: minimum number of waitstaff required in interval $t \in T$ (column "Requirement", row $t$).

Decision variables:
- $x_s \in \mathbb{Z}_+, \quad s \in T$: number of waitstaff starting their 8-hour shift at interval $s$.

Objective:
$$
\min \sum_{s \in T} x_s
$$

Constraints:
For each interval $t \in T$,
$$
\sum_{s=0}^{47} x_s \cdot \delta_{t,s} \geq r_t
$$
where
$$
\delta_{t,s} = 
\begin{cases}
1 & \text{if } (t-s) \bmod 48 \in \{0,1,\ldots,15\} \\
0 & \text{otherwise}
\end{cases}
$$
(Each shift covers 16 consecutive intervals, i.e., 8 hours.)

Variable domains:
$$
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s \in T
$$

## Data Mapping

- $T$: All 48 rows of 44.csv, indexed by source_row $0$ to $47$.
- $r_t$: "Requirement" column, table_id: file_0_view_0, row $t$.
- $x_s$: number of waitstaff starting at interval $s$.
- Each interval is 30 minutes; 16 intervals = 8 hours.
- Shift coverage: for each $t$, sum $x_s$ for all $s$ such that $s$ is in $\{t-15, ..., t\}$ modulo 48.

## Summary

Minimize total staff, assigning integer numbers of 8-hour shifts starting at each interval, so that at every interval the sum of staff on duty (those whose shift started in the last 16 intervals) meets or exceeds the required minimum from 44.csv.