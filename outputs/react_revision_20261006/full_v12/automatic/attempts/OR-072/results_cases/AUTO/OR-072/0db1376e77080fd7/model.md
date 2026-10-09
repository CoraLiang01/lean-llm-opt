## Mathematical Model

Sets:
- $T = \{1,2,\ldots,24\}$: time periods (hours in a day, from 0:00–1:00 as 1, ..., 23:00–0:00 as 24)

Parameters (from Data Mapping, table_id: file_0_view_0):
- $r_t$: required number of drivers/crew in period $t \in T$ ("Number Required" column, $t$th row)

Decision variables:
- $x_t \geq 0$, integer: number of drivers/crew starting work at the beginning of period $t$, for $t \in T$

Objective:
$$
\min \sum_{t=1}^{24} x_t
$$

Constraints (coverage: each period must have at least the required number of staff on duty):
For each $t \in T$,
$$
\sum_{k=0}^{3} x_{(t - k - 1 \bmod 24) + 1} \geq r_t
$$
where the indices wrap around modulo 24 (i.e., after period 24 comes period 1).

Variable domains:
$$
x_t \in \mathbb{Z}_{\geq 0} \quad \forall t \in T
$$

---

### Data Mapping

- $T$: All rows in table_id: file_0_view_0, column "Shift"
- $r_t$: table_id: file_0_view_0, column "Number Required", row $t$
- $x_t$: number of drivers/crew starting at period $t$ (decision variable)

---

**Note:** Each $x_t$ covers periods $t, t+1, t+2, t+3$ (modulo 24). The constraint for period $t$ sums the $x$ variables for the 4 most recent shift starts that are still on duty at $t$.