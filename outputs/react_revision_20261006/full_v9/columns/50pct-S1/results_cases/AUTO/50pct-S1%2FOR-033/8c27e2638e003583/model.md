## Mathematical Model

**Sets**
- $T$: set of 48 half-hour time intervals, indexed by $t$ (from 0 to 47), with time labels and requirements as in Data Mapping.

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement" in table_id file_0_view_0).

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \quad s = 0,1,\ldots,47$: number of waitstaff whose shift starts at interval $s$ (each shift covers 8 consecutive intervals, i.e., 4 hours).

**Objective**
$$
\min \sum_{s=0}^{47} x_s
$$

**Constraints**
For each interval $t = 0,1,\ldots,47$:
$$
\sum_{s=0}^{47} a_{t,s} x_s \geq r_t
$$
where
$$
a_{t,s} = 
\begin{cases}
1 & \text{if } t \in \{s, (s+1) \bmod 48, \ldots, (s+7) \bmod 48\} \\
0 & \text{otherwise}
\end{cases}
$$

**Variable Domains**
$$
x_s \geq 0,\quad x_s \in \mathbb{Z},\quad \forall s = 0,\ldots,47
$$

---

### Data Mapping

- $T$ and $r_t$ are defined by the 48 rows of table_id file_0_view_0, columns:
    - "Time" (interval label)
    - "Requirement" (minimum required waitstaff, integer)
- Each $x_s$ corresponds to a possible shift start at interval $s$ (row $s$ of the table).
- Each shift covers 8 consecutive intervals, wrapping around midnight as needed (modulo 48).
- All requirements $r_t$ are taken directly from the "Requirement" column for interval $t$.

**Table Reference:**  
- table_id: file_0_view_0  
- columns: "Time", "Requirement"  
- rows: 0–47 (one per half-hour interval, in order)