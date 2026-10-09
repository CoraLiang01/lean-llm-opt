## Mathematical Model

**Sets**
- $T$: set of 48 half-hour time slots, indexed by $t$ (from 0 to 47), as in 44.csv.
- $S$: set of 48 possible shift start times, indexed by $s$ (from 0 to 47). Each shift covers 16 consecutive time slots (8 hours).

**Parameters**
- $r_t$: minimum number of waitstaff required in time slot $t$ (from column "Requirement" in 44.csv, table_id: file_0_view_0, row $t$).

**Decision Variables**
- $x_s \in \mathbb{Z}_+, \forall s \in S$: number of waitstaff starting a shift at time slot $s$.

**Objective**
$$
\min \sum_{s \in S} x_s
$$

**Constraints**
For each time slot $t \in T$:
$$
\sum_{s \in S: t \in \{s, s+1, \ldots, s+15\} \bmod 48} x_s \geq r_t
$$

**Variable Domains**
$$
x_s \geq 0 \text{ and integer}, \quad \forall s \in S
$$

---

### Data Mapping

- $T$, $S$: All 48 time slots in 44.csv, table_id: file_0_view_0, column "Time", rows 0–47.
- $r_t$: Requirement for each time slot $t$, from 44.csv, table_id: file_0_view_0, column "Requirement", row $t$.
- Each $x_s$ corresponds to the number of waitstaff starting at time slot $s$ (shift start at row $s$ of 44.csv).
- Each shift covers 16 consecutive time slots, wrapping around midnight (i.e., modulo 48).

---

**Note:** For each $t$, the sum is over all $s$ such that $t$ is within the 16-slot window starting at $s$ (i.e., $t \in \{s, s+1, ..., s+15\}$ modulo 48).