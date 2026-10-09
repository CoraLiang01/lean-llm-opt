## Mathematical Model

Sets:
- $T$: set of 48 half-hour time intervals, indexed by $t$ (from 0 to 47), with time labels and requirements from 44.csv.
- $S$: set of 48 possible shift start times, indexed by $s$ (from 0 to 47). Each shift covers 16 consecutive intervals (8 hours).

Parameters:
- $r_t$: minimum number of waitstaff required in interval $t$ (from column "Requirement", table_id: file_0_view_0, row $t$).

Decision variables:
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at interval $s$.

Objective:
\[
\min \sum_{s \in S} x_s
\]

Constraints:
For each interval $t \in T$:
\[
\sum_{s \in S: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t
\]
where addition is modulo 48 (i.e., shifts wrap around midnight).

Variable domains:
\[
x_s \geq 0, \quad x_s \in \mathbb{Z}, \quad \forall s \in S
\]

---

### Data Mapping

- $T$, $S$: All 48 intervals from 44.csv, table_id: file_0_view_0, column "Time".
- $r_t$: Requirement for interval $t$, table_id: file_0_view_0, column "Requirement", row $t$.
- Each $x_s$ corresponds to the number of staff starting at time interval $s$ (table_id: file_0_view_0, row $s$).

All requirements and time intervals are mapped directly from 44.csv. Each shift covers 16 consecutive intervals (8 hours), wrapping around the 24-hour period.