**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $v_p$: Value per unit of product $p$ (from Value in file_1_view_0)
- $w_p$: Weight per unit of product $p$ (from Weight in file_1_view_0)
- $C_s$: Capacity of shelf $s$ (from Capacity in file_0_view_0)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ allocated to shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID from `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName from `file_1_view_0` (products.csv), column `ProductName`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C_s$: `file_0_view_0`, column `Capacity`, keyed by `ShelfID`