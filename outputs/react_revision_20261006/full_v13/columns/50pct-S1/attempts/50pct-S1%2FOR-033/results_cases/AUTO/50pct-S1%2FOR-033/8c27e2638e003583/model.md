## Mathematical Model

**Sets**
- $T$: set of time intervals, indexed by $t$ (from 1 to 48, each representing a 30-minute interval; see Data Mapping for exact labels)
- $S$: set of possible shift start times, indexed by $s$ (also 1 to 48, each representing a possible shift starting at interval $s$)

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement", table_id: file_0_view_0)
- $n$: number of intervals in a shift ($n = 16$, since each shift is 8 hours and each interval is 0.5 hours)

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff whose shift starts at interval $s$

**Objective**
\[
\min \sum_{s \in S} x_s
\]

**Constraints**
\[
\sum_{s \in S: t \in \text{Shift}(s)} x_s \geq r_t, \quad \forall t \in T
\]
where $\text{Shift}(s) = \{s, s+1, \ldots, s+n-1\}$ (modulo 48, i.e., wrap around after 48), meaning a shift starting at $s$ covers intervals $s$ through $s+15$ (with $s+k$ interpreted modulo 48).

\[
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
\]

**Data Mapping**
- $T$ and $S$ are both the set of 48 intervals, with labels and requirements from table_id: file_0_view_0, columns "Time" and "Requirement".
- $r_t$ is the value in "Requirement" for interval $t$.
- Each $x_s$ corresponds to the number of waitstaff starting at time interval $s$ (see "Time" in file_0_view_0).
- Each shift covers 16 consecutive intervals, wrapping around midnight as needed.

**Summary**
Minimize the total number of waitstaff scheduled, ensuring that for every interval, the sum of staff present (i.e., those whose 8-hour shift covers that interval) meets or exceeds the required minimum. Each staff member works a continuous 8-hour (16-interval) shift, and shifts can start at any interval. All variables are nonnegative integers.