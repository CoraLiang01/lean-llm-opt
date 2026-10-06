**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $v_p$: Value of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: Weight of product $p$ (file_1_view_0, column Weight, key ProductName)
- $C_s$: Capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)

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

2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All ShelfID from file_0_view_0 (capacity.csv), column ShelfID
- $P$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $C_s$: file_0_view_0, column Capacity, key ShelfID
- $x_{s,p}$: Decision variable for each $(s,p) \in S \times P$