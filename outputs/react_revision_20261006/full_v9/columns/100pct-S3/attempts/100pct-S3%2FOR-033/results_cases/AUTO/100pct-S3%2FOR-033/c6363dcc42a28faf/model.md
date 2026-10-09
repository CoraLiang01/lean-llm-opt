## Mathematical Model

**Sets**
- $T$: set of time intervals (indexed by $t$), as in 44.csv, $|T|=48$
- $S$: set of possible shift start times (indexed by $s$), $S = T$ (one possible shift start per interval)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from 44.csv, column "Requirement", table_id: file_0_view_0)
- $n$: number of intervals per shift ($n=16$, since 8 hours / 0.5 hour per interval)
- $|T|$: total number of intervals in a day ($|T|=48$)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For each interval $t \in T$:
$$
\sum_{k=0}^{n-1} x_{(t - k) \bmod |T|} \geq r_t
$$

**Variable domains**
$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals in 44.csv, column "Time", table_id: file_0_view_0
- $r_t$: 44.csv, column "Requirement", table_id: file_0_view_0, row $t$
- $x_s$: number of waitstaff starting at time interval $s$ (decision variable)
- $n=16$: Each shift covers 8 hours = 16 consecutive 30-minute intervals
- The sum in the constraint uses modular arithmetic to wrap around midnight

**Table Reference:** All parameters are mapped to 44.csv, table_id: file_0_view_0, columns "Time" and "Requirement".