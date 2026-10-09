## Mathematical Model

**Sets**
- $S = \{1,2,\ldots,48\}$: Index set of half-hour time slots (from 2:00am–2:30am as $s=1$ to 1:30am–2:00am as $s=48$), as in 44.csv.

**Parameters**
- $r_s$: Minimum number of waitstaff required in time slot $s \in S$ (from column "Requirement" in 44.csv, table_id: file_0_view_0).

**Decision Variables**
- $x_t \in \mathbb{Z}_+, \ t \in S$: Number of waitstaff whose shift starts at time slot $t$ (each works 8 consecutive hours, i.e., 16 consecutive half-hour slots).

**Objective**
\[
\min \sum_{t=1}^{48} x_t
\]

**Constraints**
\[
\forall s \in S: \quad \sum_{t=1}^{48} a_{t,s} \, x_t \geq r_s
\]
where
\[
a_{t,s} = 
\begin{cases}
1 & \text{if time slot } s \text{ is covered by a shift starting at } t \\
0 & \text{otherwise}
\end{cases}
\]
and a shift starting at $t$ covers time slots $t, t+1, \ldots, t+15$ (modulo 48, i.e., wrap around after 48).

**Variable Domains**
\[
x_t \geq 0, \quad x_t \in \mathbb{Z}, \quad \forall t \in S
\]

---

### Data Mapping

- $S$: All 48 time slots from column "Time" in 44.csv (table_id: file_0_view_0).
- $r_s$: "Requirement" column, row $s$ (source_row $s-1$), table_id: file_0_view_0.
- $x_t$: Decision variable for number of waitstaff starting at time slot $t$.
- $a_{t,s}$: $1$ if $s \in \{t, t+1, ..., t+15\}$ modulo 48, $0$ otherwise.

**Note:** All indices $t, s$ are modulo 48 (i.e., after 48 comes 1). Each $x_t$ represents a shift starting at time slot $t$ and covering 16 consecutive half-hour slots (8 hours).