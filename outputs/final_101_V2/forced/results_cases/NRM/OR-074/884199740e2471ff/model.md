#### Abstract Mathematical Model

**Index Sets:**
- $T$: set of all time slots in the day (from the "Time" column in 44.csv), indexed by $t$.

**Parameters:**
- $r_t$: minimum number of waitstaff required in time slot $t$ (from the "Requirement" column in 44.csv).
- $n$: number of consecutive time slots covered by one waitstaff shift ($n = 16$, since each slot is 30 minutes and each shift is 8 hours).

**Decision Variables:**
- $x_s \in \mathbb{Z}_+, \quad \forall s \in T$: number of waitstaff whose shift starts at time slot $s$.

**Objective:**
\[
\min \sum_{s \in T} x_s
\]

**Constraints:**
\[
\sum_{s \in S_t} x_s \geq r_t, \quad \forall t \in T
\]
where $S_t = \{ s \in T : t \text{ is covered by the 8-hour shift starting at } s \}$, i.e., $S_t = \{ s \in T : t \in [s, s+n-1] \text{ (modulo } |T|) \}$.

\[
x_s \in \mathbb{Z}_+, \quad \forall s \in T
\]

---

#### Data Mapping

- Table: 44.csv
    - Index set $T$: column "Time", table_id: file_0_view_0
    - Parameter $r_t$: column "Requirement", table_id: file_0_view_0

- Each $x_s$ corresponds to a possible shift start at time slot $s \in T$.

- Each shift covers $n=16$ consecutive time slots (8 hours, 30 minutes per slot), wrapping around midnight as needed.