## Mathematical Model

**Sets**
- $T$: set of time intervals, indexed by $t$ (from 1 to 48, each representing a 30-minute interval; see Data Mapping for exact labels)
- $S$: set of possible shift start times, indexed by $s$ (also 1 to 48, one for each interval)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from Requirement column, table_id: file_0_view_0)
- Each shift covers 16 consecutive intervals (8 hours = 16 × 30 minutes)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$

**Objective**
\[
\min \sum_{s \in S} x_s
\]

**Constraints**
\[
\sum_{s \in S: t \in \text{Shift}(s)} x_s \geq r_t, \quad \forall t \in T
\]
where $\text{Shift}(s) = \{s, s+1, \ldots, s+15\}$ (modulo 48, i.e., wrap around after 48 to 1), meaning a shift starting at $s$ covers intervals $s$ through $s+15$ (with $s+k$ interpreted modulo 48).

\[
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
\]

---

### Data Mapping

- $T$, $S$: All 48 intervals from the "Time" column in table_id: file_0_view_0.
- $r_t$: "Requirement" column, table_id: file_0_view_0, for each $t$.
- Each $x_s$ corresponds to the number of waitstaff starting at time interval $s$ (from the same set as $T$).
- Each shift covers 16 consecutive intervals, wrapping around midnight as needed.

**Table Reference:**  
- table_id: file_0_view_0  
- Columns: "Time" (interval label), "Requirement" (minimum waitstaff needed)

---

**Summary:**  
Minimize the total number of waitstaff scheduled, ensuring that for every interval, the sum of staff on duty (i.e., those whose 8-hour shift covers that interval) meets or exceeds the required minimum. Each shift can start at any interval, and shifts wrap around the 24-hour period.