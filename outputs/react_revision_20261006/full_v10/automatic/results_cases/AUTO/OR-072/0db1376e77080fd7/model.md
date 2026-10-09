## Mathematical Model

Sets:
- $T = \{1,2,\ldots,24\}$: set of time periods (hours in a day, as indexed in the data)
- For each $t \in T$, let $r_t$ be the required number of drivers and crew members in period $t$ (from Data Mapping).

Parameters:
- $r_t$: Number Required in period $t$ (from 42.csv, column "Number Required", table_id: file_0_view_0)

Decision variables:
- $x_s \geq 0$, integer: number of drivers and crew members starting work at the beginning of period $s$, for $s \in T$

Objective:
$$
\min \sum_{s=1}^{24} x_s
$$

Constraints:
For each time period $t \in T$:
$$
\sum_{k=0}^{3} x_{(t - k - 1 \bmod 24) + 1} \geq r_t
$$
where the indices are taken modulo 24 (i.e., after period 24 comes period 1), so that for each period $t$, the sum covers all staff who started in the current or previous 3 periods and are still on duty.

Variable domains:
$$
x_s \in \mathbb{Z}_{\geq 0} \quad \forall s \in T
$$

### Data Mapping

- $T$: All rows in 42.csv, column "Shift" (table_id: file_0_view_0)
- $r_t$: 42.csv, column "Number Required", table_id: file_0_view_0, row with Shift $t$
- $x_s$: Number of staff starting at period $s$ (decision variable, not in data)

All indices and parameters are mapped directly from the current 42.csv as described.