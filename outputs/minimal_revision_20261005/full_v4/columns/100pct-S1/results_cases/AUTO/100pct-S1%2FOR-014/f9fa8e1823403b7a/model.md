**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from `file_0_view_0`, column `ShelfID`)
- $P$: Set of products, indexed by $p$ (from `file_1_view_0`, column `ProductName`)

**Parameters:**
- $c_s$: Capacity of shelf $s$ (from `file_0_view_0`, column `Capacity`)
- $v_p$: Value of product $p$ (from `file_1_view_0`, column `Value`)
- $w_p$: Weight of product $p$ (from `file_1_view_0`, column `Weight`)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

- $S$: All values in `file_0_view_0`, column `ShelfID`
- $P$: All values in `file_1_view_0`, column `ProductName`
- $c_s$: `file_0_view_0`, column `Capacity`, keyed by `ShelfID`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`