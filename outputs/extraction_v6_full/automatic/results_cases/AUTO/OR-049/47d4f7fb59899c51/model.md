## Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0.ProductName)

**Parameters:**
- $C_s$: capacity of shelf $s$ (from file_0_view_0.Capacity)
- $v_p$: value of product $p$ (from file_1_view_0.Value)
- $w_p$: weight of product $p$ (from file_1_view_0.Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

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

- $S$ (Shelves): file_0_view_0.ShelfID
- $C_s$: file_0_view_0.Capacity, keyed by file_0_view_0.ShelfID
- $P$ (Products): file_1_view_0.ProductName
- $v_p$: file_1_view_0.Value, keyed by file_1_view_0.ProductName
- $w_p$: file_1_view_0.Weight, keyed by file_1_view_0.ProductName

All parameters are mapped directly from the corresponding columns and keys in the retrieved CSV files.