**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: Capacity of shelf $s$ (file_0_view_0, column: Capacity, key: ShelfID)
- $v_p$: Value per unit of product $p$ (file_1_view_0, column: Value, key: ProductName)
- $w_p$: Weight per unit of product $p$ (file_1_view_0, column: Weight, key: ProductName)

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
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]

2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All ShelfID from `file_0_view_0`, column `ShelfID`
- $P$: All ProductName from `file_1_view_0`, column `ProductName`
- $c_s$: `file_0_view_0`, column `Capacity`, keyed by `ShelfID`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`