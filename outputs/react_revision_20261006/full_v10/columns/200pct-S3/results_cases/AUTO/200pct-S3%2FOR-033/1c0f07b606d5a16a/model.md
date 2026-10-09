Let $T$ be the set of 48 half-hour intervals in the day, indexed in source order by $t=0,\ldots,47$, with required minimum staff $r_t$ from column "Requirement" in table_id file_0_view_0. Let $x_s$ be the number of waitstaff starting work at interval $s$ ($s=0,\ldots,47$). Each waitstaff works 16 consecutive intervals (8 hours).

Sets:
- $T = \{0,1,\ldots,47\}$ (half-hour intervals, source_row in file_0_view_0)

Parameters (from Data Mapping):
- $r_t$ = Requirement at interval $t$ (file_0_view_0, column "Requirement", row $t$)

Decision variables:
- $x_s \in \mathbb{Z}_{\geq 0}$: number of waitstaff starting at interval $s$

Objective:
Minimize total number of waitstaff:
$$
\min \sum_{s=0}^{47} x_s
$$

Constraints:
For each interval $t \in T$,
$$
\sum_{s=0}^{47} a_{t,s} x_s \geq r_t
$$
where
$$
a_{t,s} = \begin{cases}
1 & \text{if } (t-s) \bmod 48 \in \{0,1,\ldots,15\} \\
0 & \text{otherwise}
\end{cases}
$$
That is, $x_s$ covers intervals $s, s+1, \ldots, s+15$ (modulo 48).

Variable domains:
$$
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s \in T
$$

Data Mapping:
- $T$: file_0_view_0, all rows, column "Time"
- $r_t$: file_0_view_0, column "Requirement", row $t$
- $x_s$: decision variable, start at interval $s$ (corresponds to file_0_view_0, row $s$)

All parameters and indices are defined by the current data.