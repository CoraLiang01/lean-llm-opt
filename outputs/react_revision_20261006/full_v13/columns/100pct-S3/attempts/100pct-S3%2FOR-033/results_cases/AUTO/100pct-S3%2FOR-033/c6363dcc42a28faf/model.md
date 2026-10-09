## Mathematical Model

**Sets**
- $T$: set of all time intervals, indexed by $t$ (from 1 to 48, each representing a 30-minute interval; see Data Mapping for exact labels)
- $S$: set of possible shift start times, indexed by $s$ (also 1 to 48, each corresponding to a time interval)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from Requirement column, table_id: file_0_view_0)
- $L$: length of a shift in intervals ($L = 16$, since 8 hours / 0.5 hour per interval)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
$$
\sum_{s \in S: t \in \{s, s+1, \ldots, s+L-1\} \bmod 48} x_s \geq r_t, \quad \forall t \in T
$$

$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

**Data Mapping**
- $T$ and $S$ are both the set of 48 time intervals from the "Time" column in table_id: file_0_view_0.
- $r_t$ is the "Requirement" value for interval $t$ from table_id: file_0_view_0.
- Each $x_s$ corresponds to the number of waitstaff whose shift starts at the time in row $s$ of table_id: file_0_view_0.
- Each shift covers 16 consecutive intervals, wrapping around midnight as needed (i.e., modulo 48 arithmetic).

**Notes**
- For each interval $t$, the sum is over all $s$ such that a shift starting at $s$ covers $t$ (i.e., $t$ is in $[s, s+L-1]$ modulo 48).
- All data is mapped directly from table_id: file_0_view_0, columns "Time" and "Requirement".