### Mathematical Model

Let $I$ be the set of vehicle types from file_1_view_0.ProductName.

**Parameters:**
- $p_i$: profit per unit of vehicle $i$ (file_1_view_0.Value)
- $w_i$: weight (inventory space) per unit of vehicle $i$ (file_1_view_0.Weight)
- $C$: total inventory capacity (file_0_view_0.Capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: file_1_view_0.ProductName
- $p_i$: file_1_view_0.Value (for each $i$)
- $w_i$: file_1_view_0.Weight (for each $i$)
- $C$: file_0_view_0.Capacity
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, nonnegative integer)