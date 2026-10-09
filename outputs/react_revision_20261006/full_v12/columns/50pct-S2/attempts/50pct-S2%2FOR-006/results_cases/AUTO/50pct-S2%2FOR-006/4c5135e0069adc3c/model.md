#### Mathematical Model

Let $I$ be the set of vehicle types from file_1_view_0.ProductName.

**Parameters:**
- $v_i$: benefit coefficient of vehicle type $i$ (file_1_view_0.Value)
- $w_i$: inventory weight of vehicle type $i$ (file_1_view_0.Weight)
- $C$: total inventory capacity (file_0_view_0.Capacity)

**Decision Variables:**
- $x_i$: number of units of vehicle type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C$: file_0_view_0.Capacity (scalar)
- $x_i$: integer, for each $i \in I$