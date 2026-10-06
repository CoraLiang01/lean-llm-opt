## Abstract Mathematical Model

**Sets:**
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight of product $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]
2. **Integrality and Nonnegativity:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

## Data Mapping

- **file_0_view_0 (capacity.csv):**
  - Shelf index $s$: ShelfID
  - Shelf capacity $C_s$: Capacity

- **file_1_view_0 (products.csv):**
  - Product index $p$: ProductName
  - Product value $v_p$: Value
  - Product weight $w_p$: Weight

All indices and parameters must be mapped exactly as above, preserving the original file and column names.