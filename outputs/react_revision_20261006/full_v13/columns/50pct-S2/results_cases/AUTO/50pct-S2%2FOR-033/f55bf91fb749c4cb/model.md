## Mathematical Model

**Sets**
- $T$: set of all time intervals, indexed by $t$ (from the "Time" column in 44.csv, $|T|=48$)
- $S$: set of all possible shift start times, indexed by $s$ (one for each interval, $|S|=48$)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from "Requirement" in 44.csv, table_id: file_0_view_0)
- $L$: number of consecutive intervals in an 8-hour shift ($L = 16$, since each interval is 30 minutes and 8 hours = 16 intervals)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For every interval $t \in T$:
$$
\sum_{s \in S: t \in \{s, s+1, \ldots, s+L-1\} \pmod{|T|}} x_s \geq r_t
$$

**Variable Domains**
$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 intervals from the "Time" column in 44.csv (table_id: file_0_view_0, column: "Time")
- $r_t$: "Requirement" column in 44.csv (table_id: file_0_view_0, column: "Requirement")
- $L = 16$: Each shift covers 16 consecutive intervals (8 hours × 2 intervals/hour)
- $x_s$: Number of waitstaff starting at interval $s$ (decision variable for each $s \in S$)

**Coverage logic:** For each interval $t$, the sum is over all $x_s$ where a shift starting at $s$ covers $t$, i.e., $t$ is in the set $\{s, s+1, ..., s+L-1\}$ modulo 48 (to wrap around midnight).

**Source Table:** file_0_view_0 (44.csv), columns "Time" and "Requirement".