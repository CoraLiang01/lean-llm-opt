## Mathematical Model

**Sets**
- $T = \{1, 2, \ldots, 48\}$: Index set of 30-minute time intervals (from 2:00am–2:30am, ..., 1:30am–2:00am), as in 44.csv.
- $S = \{1, 2, \ldots, 48\}$: Index set of possible shift start times (one for each interval).

**Parameters** (from 44.csv, table_id: file_0_view_0)
- $r_t$: Minimum number of waitstaff required in interval $t \in T$; $r_t =$ Requirement column, row $t$.
- Each shift covers 16 consecutive intervals (8 hours = 16 × 30min).

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: Number of waitstaff whose shift starts at interval $s$.

**Objective**
\[
\min \sum_{s \in S} x_s
\]

**Constraints**
\[
\forall t \in T: \quad \sum_{s \in S: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t
\]
where addition is modulo 48 (i.e., after interval 48 comes interval 1).

**Variable Domains**
\[
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
\]

---

### Data Mapping

- $T$, $S$: All 48 intervals, as indexed by the rows of 44.csv (table_id: file_0_view_0, column "Time").
- $r_t$: file_0_view_0, column "Requirement", row $t$.
- Each $x_s$ corresponds to the number of waitstaff starting at time interval $s$ (file_0_view_0, row $s$).
- Each shift covers 16 consecutive intervals, wrapping around midnight (modulo 48).

---

**Summary:**  
Minimize the total number of waitstaff, assigning integer numbers of staff to each possible 8-hour shift start time, so that at every 30-minute interval, the sum of staff on duty (i.e., those whose shift covers that interval) meets or exceeds the required minimum from the data. All data is mapped directly from 44.csv (file_0_view_0).