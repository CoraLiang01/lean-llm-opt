## Abstract Mathematical Model

### Index Sets
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0.ProductName)

### Parameters
- $C_s$: Capacity of shelf $s$ (from file_0_view_0.Capacity)
- $v_p$: Value of product $p$ (from file_1_view_0.Value)
- $w_p$: Weight of product $p$ (from file_1_view_0.Weight)

### Decision Variables
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

### Objective
Maximize total value of products allocated to shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
$$

### Constraints

1. **Shelf Capacity Constraints** (for each shelf $s \in S$):
   $$
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
   $$

2. **Nonnegativity and Integrality**:
   $$
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   $$

---

### Data Mapping

- $S$ (Shelves): file_0_view_0.ShelfID
- $C_s$: file_0_view_0.Capacity (shelf $s$)
- $P$ (Products): file_1_view_0.ProductName
- $v_p$: file_1_view_0.Value (product $p$)
- $w_p$: file_1_view_0.Weight (product $p$)