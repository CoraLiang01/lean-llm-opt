#### Abstract Mathematical Model

Let  
- $T$ = set of time periods, indexed by $t$ (from 44.csv, column "Time")  
- $R_t$ = minimum number of waitstaff required in time period $t$ (from 44.csv, column "Requirement")  
- $x_s$ = number of waitstaff whose shift starts at time period $s$ ($s \in T$), decision variable

Each waitstaff works a continuous 8-hour shift (i.e., 16 consecutive half-hour periods). The schedule is cyclic over 24 hours (48 half-hour periods).

**Objective:**  
Minimize the total number of waitstaff scheduled:
$$
\min \sum_{s \in T} x_s
$$

**Constraints:**  
For every time period $t \in T$, the total number of waitstaff on duty must meet or exceed the requirement:
$$
\sum_{s \in T: t \in \text{Shift}(s)} x_s \geq R_t, \quad \forall t \in T
$$
where $\text{Shift}(s)$ is the set of 16 consecutive time periods starting at $s$ (wrapping around midnight as needed).

**Variable domains:**  
$$
x_s \in \mathbb{Z}_{\geq 0}, \quad \forall s \in T
$$

---

#### Data Mapping

- $T$ (time periods): 44.csv, column "Time", table_id: file_0_view_0
- $R_t$ (requirements): 44.csv, column "Requirement", table_id: file_0_view_0, key: "Time"
- $x_s$: number of waitstaff starting at time period $s$ (indexed by "Time" from 44.csv)

Each $x_s$ is associated with a unique "Time" value from 44.csv. For each $t$, the sum is over all $x_s$ such that $t$ is within the 16-period shift starting at $s$ (modulo 48).

No other columns or files are used. All requirements and time periods are included as given, in original file order.