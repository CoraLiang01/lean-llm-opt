#### Abstract Mathematical Model

**Sets:**
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight of product $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ to place on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

#### Data Mapping

- $S$ (Shelves): file_0_view_0, column ShelfID
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $P$ (Products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

All data is used in original file and row order, with explicit business identifiers.