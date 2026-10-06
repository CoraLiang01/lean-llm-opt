#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves (indexed by $s$), from file_0_view_0.ShelfID
- $P$: set of products (indexed by $p$), from file_1_view_0.ProductName

**Parameters:**
- $C_s$: capacity of shelf $s$, from file_0_view_0.Capacity
- $v_p$: value per unit of product $p$, from file_1_view_0.Value
- $w_p$: weight per unit of product $p$, from file_1_view_0.Weight

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]
- Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0.ShelfID
- $C_s$: file_0_view_0.Capacity (matched by ShelfID)
- $P$ (products): file_1_view_0.ProductName
- $v_p$: file_1_view_0.Value (matched by ProductName)
- $w_p$: file_1_view_0.Weight (matched by ProductName)