## Mathematical Model

**Sets**
- $T$: set of all 48 half-hour time intervals in the day, indexed by $t$ (from 1 to 48), as given in 44.csv.
- $S$: set of all possible shift start times, $S = T$ (since a shift can start at any interval).

**Parameters**
- $r_t$: minimum number of waitstaff required in interval $t \in T$ (from column "Requirement" in 44.csv, table_id: file_0_view_0).
- $L$: number of consecutive intervals in a shift ($L = 16$, since each shift is 8 hours and each interval is 0.5 hours: $8 \times 2 = 16$).

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \quad \forall s \in S$: number of waitstaff whose shift starts at interval $s$.

**Objective**
\[
\min \sum_{s \in S} x_s
\]

**Constraints**
\[
\sum_{s \in S: \, t \in \{s, s+1, \ldots, s+L-1\} \bmod 48} x_s \geq r_t, \quad \forall t \in T
\]
where addition is modulo 48, i.e., after interval 48 comes interval 1 (to cover wrap-around shifts crossing midnight).

\[
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
\]

---

### Data Mapping

- $T$, $S$: All 48 intervals from column "Time" in 44.csv (table_id: file_0_view_0).
- $r_t$: "Requirement" column in 44.csv (table_id: file_0_view_0), mapped to each $t$.
- $L = 16$ (since each shift is 8 hours, each interval is 0.5 hours).
- $x_s$: Number of waitstaff starting at interval $s$ (decision variable for each $s \in S$).

**Source Table:** 44.csv (table_id: file_0_view_0), columns "Time", "Requirement".