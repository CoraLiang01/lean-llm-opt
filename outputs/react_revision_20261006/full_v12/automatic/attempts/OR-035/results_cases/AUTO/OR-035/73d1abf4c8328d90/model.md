#### Mathematical Model

Let $I$ be the set of bread types from file_1_view_0[ProductName].

**Decision variables:**
- $x_i$: number of units of bread type $i \in I$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Parameters:**
- $v_i$: expected profit per unit of bread $i$ (file_1_view_0[Value])
- $w_i$: weight (storage space required) per unit of bread $i$ (file_1_view_0[Weight])
- $C$: total storage capacity (file_0_view_0[Capacity])

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: file_1_view_0[ProductName]
- $v_i$: file_1_view_0[Value] for bread $i$
- $w_i$: file_1_view_0[Weight] for bread $i$
- $C$: file_0_view_0[Capacity]
- $x_i$: number of units to order of bread $i$ (decision variable, integer, $\geq 0$)