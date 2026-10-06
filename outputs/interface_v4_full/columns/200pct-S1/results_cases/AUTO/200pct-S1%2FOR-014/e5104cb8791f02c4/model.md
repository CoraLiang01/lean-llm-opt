#### Abstract Mathematical Model

**Sets:**
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $v_p$: Value of one unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight of one unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Shelf capacity for each $s \in S$:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]
- Integrality and nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- **file_0_view_0 (capacity.csv):**
    - Shelf index: ShelfID
    - Shelf capacity: Capacity

- **file_1_view_0 (products.csv):**
    - Product index: ProductName
    - Product value: Value
    - Product weight: Weight

All sets and parameters are defined by the full, unsorted, source order of the above columns. Every shelf and product record is included.