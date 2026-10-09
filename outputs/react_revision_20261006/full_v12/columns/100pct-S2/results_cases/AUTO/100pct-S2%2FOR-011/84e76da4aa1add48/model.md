#### Mathematical Model

Let $I$ be the set of products from file_1_view_0[ProductName].

**Decision variables:**
- For each $i \in I$, $x_i \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ to order each day.

**Parameters:**
- $v_i$: value (benefit) per unit of product $i$ (from file_1_view_0[Value])
- $w_i$: weight (stock space required) per unit of product $i$ (from file_1_view_0[Weight])
- $C$: overall stock capacity (from file_0_view_0[Capacity])

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
- $v_i$: file_1_view_0[Value] for product $i$
- $w_i$: file_1_view_0[Weight] for product $i$
- $C$: file_0_view_0[Capacity]
- $x_i$: number of units of product $i$ to order each day (decision variable)