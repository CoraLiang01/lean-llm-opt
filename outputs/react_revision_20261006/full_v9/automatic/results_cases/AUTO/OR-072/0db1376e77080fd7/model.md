## Mathematical Model

**Sets**

- $T = \{1,2,\ldots,24\}$: set of time periods (hours in a day, as indexed in the data)
- Let $r_t$ = required number of drivers/crew in period $t$ (from data, see Data Mapping)

**Parameters**

- $r_t$: Number Required in period $t$ (from 42.csv, column "Number Required", row $t$)

**Decision Variables**

- $x_t \in \mathbb{Z}_+, \forall t \in T$: number of drivers/crew members starting work at the beginning of period $t$

**Objective**

$$
\min \sum_{t=1}^{24} x_t
$$

**Constraints**

For each period $t \in T$:
$$
\sum_{k=0}^{3} x_{(t - k - 1 \bmod 24) + 1} \geq r_t
\qquad \forall t \in T
$$

where the indices are taken modulo 24 (i.e., after period 24 comes period 1), so that for each period $t$, the sum covers all drivers/crew who started in the current or previous 3 periods and are still on duty.

**Variable Domains**

$$
x_t \geq 0,\quad x_t \in \mathbb{Z},\quad \forall t \in T
$$

---

### Data Mapping

- $T$: All rows in 42.csv, column "Shift" (1 to 24)
- $r_t$: 42.csv, column "Number Required", row with "Shift" = $t$
- $x_t$: Number of drivers/crew starting at the beginning of period $t$ (decision variable)

---

**Summary:**  
Minimize the total number of drivers/crew assigned, ensuring that in every period, the sum of those who started in the current and previous 3 periods (i.e., are still on duty for their 4-hour shift) meets or exceeds the required number for that period. All variables are nonnegative integers.