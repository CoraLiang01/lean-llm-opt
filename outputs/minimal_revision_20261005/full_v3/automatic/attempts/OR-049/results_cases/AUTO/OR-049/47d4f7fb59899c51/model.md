**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight of product $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
   \]
2. **Integrality and Nonnegativity:**  
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$ (shelves): file_0_view_0, column ShelfID
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

**Variables:**
- $x_{s,p}$: Number of units of product $p$ on shelf $s$ (indexed by ShelfID and ProductName)

---

**Summary:**  
Maximize total product value placed on shelves, subject to each shelf's capacity, using integer product allocations. All parameters and indices are mapped directly to the supplied data columns.