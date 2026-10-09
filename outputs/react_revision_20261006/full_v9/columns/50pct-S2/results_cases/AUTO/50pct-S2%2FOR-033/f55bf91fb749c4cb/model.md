## Mathematical Model

**Sets**
- $S = \{1,2,\ldots,48\}$: Set of half-hour time slots (indexed in order as in 44.csv)
- $W = \{1,2,\ldots,48\}$: Set of possible shift start times (one per time slot; each shift covers 8 consecutive slots)

**Parameters**
- $r_s$: Minimum number of waitstaff required in time slot $s \in S$ (from column "Requirement", table_id: file_0_view_0, row $s-1$)
- Each shift $w \in W$ covers time slots $C(w) = \{w, w+1, \ldots, w+15\}$ (modulo 48, i.e., wrap around after slot 48 to slot 1)

**Decision Variables**
- $x_w \in \mathbb{Z}_+, \quad \forall w \in W$: Number of waitstaff starting a shift at time slot $w$

**Objective**
$$
\min \sum_{w \in W} x_w
$$

**Constraints**
$$
\sum_{w: s \in C(w)} x_w \geq r_s, \quad \forall s \in S
$$

$$
x_w \geq 0 \text{ and integer}, \quad \forall w \in W
$$

**Data Mapping**
- $r_s$ is mapped to "Requirement" in 44.csv (table_id: file_0_view_0, column "Requirement", row $s-1$)
- $S$ and $W$ both correspond to the 48 time slots in 44.csv (table_id: file_0_view_0, column "Time")
- Each $x_w$ corresponds to the number of waitstaff starting at the time slot indexed by $w$ in 44.csv

**Coverage Definition**
- For each $w \in W$, $C(w) = \{w, w+1, ..., w+15\}$ modulo 48 (i.e., after 48 comes 1), representing the 8-hour (16-slot) continuous shift starting at slot $w$.

**Summary**
- The model minimizes the total number of waitstaff scheduled, ensuring that at every half-hour time slot, the sum of all waitstaff on duty (i.e., those whose 8-hour shift covers that slot) meets or exceeds the required minimum. All variables and parameters are mapped directly to the data in 44.csv.