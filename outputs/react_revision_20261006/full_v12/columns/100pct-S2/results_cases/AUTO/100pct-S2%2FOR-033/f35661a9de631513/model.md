Let $T = \{1,2,\ldots,48\}$ index the 48 half-hour intervals in the day, in the order given in 44.csv. Let $R_t$ be the required minimum number of waitstaff in interval $t$ (from column "Requirement", table_id: file_0_view_0, row $t-1$). Let $x_s$ be the number of waitstaff starting work at interval $s$ ($s \in T$).

Minimize
$$
\sum_{s=1}^{48} x_s
$$

subject to, for all $t \in T$,
$$
\sum_{k=0}^{15} x_{(t - k - 1 \bmod 48) + 1} \geq R_t
$$

and
$$
x_s \geq 0,\quad x_s \in \mathbb{Z} \quad \forall s \in T
$$

**Data Mapping:**
- $T$: All 48 rows of 44.csv, column "Time" (table_id: file_0_view_0)
- $R_t$: 44.csv, column "Requirement", row $t-1$ (table_id: file_0_view_0)
- $x_s$: Number of waitstaff starting at interval $s$ (decision variable, integer, nonnegative)
- Each $x_s$ covers intervals $s, s+1, ..., s+15$ (modulo 48), i.e., 8 hours = 16 intervals

**Explanation of coverage:** Each waitstaff works 16 consecutive intervals (8 hours), so for each interval $t$, the sum counts all staff who started in the previous 16 intervals (including $t$), wrapping around midnight.