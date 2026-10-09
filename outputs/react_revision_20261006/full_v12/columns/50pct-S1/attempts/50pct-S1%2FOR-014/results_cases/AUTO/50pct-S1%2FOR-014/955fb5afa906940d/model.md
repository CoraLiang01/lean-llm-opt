### Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0.ProductName)

**Parameters:**
- $C_s$: capacity of shelf $s$ (from file_0_view_0.Capacity)
- $v_p$: value per unit of product $p$ (from file_1_view_0.Value)
- $w_p$: weight per unit of product $p$ (from file_1_view_0.Weight)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
- Shelf capacity for each shelf $s$:
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]
- Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

### Data Mapping

- $S$: All values of ShelfID from file_0_view_0 (capacity.csv)
- $P$: All values of ProductName from file_1_view_0 (products.csv)
- $C_s$: file_0_view_0.Capacity, keyed by ShelfID
- $v_p$: file_1_view_0.Value, keyed by ProductName
- $w_p$: file_1_view_0.Weight, keyed by ProductName
- $x_{sp}$: Decision variable for units of product $p$ on shelf $s$ (indexed by ShelfID and ProductName)