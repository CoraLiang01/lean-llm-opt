Let $T$ be the set of 48 half-hour time intervals in the day, indexed in order as $t=1,\ldots,48$, with requirement $r_t$ for each $t$ from 44.csv (column "Requirement", table_id: file_0_view_0). Let $x_s$ be the number of waitstaff starting work at interval $s$ ($s=1,\ldots,48$). Each waitstaff works 8 consecutive hours (16 intervals).

Minimize
$$
\sum_{s=1}^{48} x_s
$$

Subject to, for all $t=1,\ldots,48$:
$$
\sum_{s=1}^{48} a_{s,t} x_s \geq r_t
$$
where $a_{s,t} = 1$ if a staff member starting at $s$ is working during interval $t$, i.e., if $t \in \{s, s+1, \ldots, s+15\}$ modulo 48 (wrap-around), and $a_{s,t} = 0$ otherwise.

Variable domains:
$$
x_s \geq 0 \text{ and integer} \quad \forall s=1,\ldots,48
$$

Data Mapping:
- $T$: All 48 rows of 44.csv, table_id: file_0_view_0, column "Time"
- $r_t$: 44.csv, table_id: file_0_view_0, column "Requirement", row $t$
- $x_s$: decision variable, number of staff starting at interval $s$
- $a_{s,t}$: defined as above, with wrap-around modulo 48

This model ensures at every interval $t$ the sum of all staff present (those who started in the previous 16 intervals) meets or exceeds the required minimum. The objective is to minimize the total number of staff scheduled.