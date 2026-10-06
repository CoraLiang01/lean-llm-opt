#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $v_p$: value of one unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: weight of one unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$: capacity of shelf $s$ (from file_0_view_0, column Capacity)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Constraints:**
- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
\]
- Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{s,p} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

All records and columns from both files are included, preserving source order and identifiers.