Let $T = \{1,2,\ldots,48\}$ index the 48 half-hour intervals in the day, in the order given in 44.csv. Let $R_t$ be the required minimum number of waitstaff in interval $t \in T$, as given by the Requirement column of 44.csv. Let $x_s$ be the number of waitstaff whose shift starts at interval $s \in T$ (decision variables, integer, $x_s \geq 0$). Each shift covers 8 consecutive intervals (4 hours), wrapping around midnight.

Objective:
Minimize $\sum_{s=1}^{48} x_s$

Subject to:
For all $t \in T$,
$$
\sum_{s=1}^{48} a_{t,s} x_s \geq R_t
$$
where $a_{t,s} = 1$ if interval $t$ is covered by a shift starting at $s$, i.e., if $t \in \{s, s+1, \ldots, s+15\}$ modulo 48 (since each shift is 8 hours = 16 intervals), and $a_{t,s} = 0$ otherwise.

Variable domains:
$$
x_s \in \mathbb{Z}_{\geq 0} \quad \forall s \in T
$$

Data Mapping:
- $T$: All 48 rows of 44.csv, indexed in file order (file_0_view_0, column "Time")
- $R_t$: file_0_view_0, column "Requirement", row $t-1$
- $x_s$: number of waitstaff starting at time interval $s$ (decision variable)
- $a_{t,s}$: 1 if $t$ is in $\{s, s+1, ..., s+15\}$ modulo 48, 0 otherwise

Minimize total number of waitstaff, ensuring at every interval the sum of all on-duty staff (from all shifts covering that interval) meets or exceeds the required minimum.