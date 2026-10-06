#### Abstract Mathematical Model

**Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0.ProductName)

**Parameters:**
- $c_s$: capacity of shelf $s$ (from file_0_view_0.Capacity)
- $v_p$: value of product $p$ (from file_1_view_0.Value)
- $w_p$: weight of product $p$ (from file_1_view_0.Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s
\]
- Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0.ShelfID
- $P$ (products): file_1_view_0.ProductName
- $c_s$: file_0_view_0.Capacity, indexed by ShelfID
- $v_p$: file_1_view_0.Value, indexed by ProductName
- $w_p$: file_1_view_0.Weight, indexed by ProductName

Each $x_{sp}$ is defined for every combination of shelf $s$ (file_0_view_0.ShelfID) and product $p$ (file_1_view_0.ProductName). All parameters are mapped directly from the corresponding columns and business IDs in the source files.