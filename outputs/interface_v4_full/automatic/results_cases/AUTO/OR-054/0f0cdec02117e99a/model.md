#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $i$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $j$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_i$: capacity of shelf $i$ (from file_0_view_0, column Capacity)
- $v_j$: value per unit of product $j$ (from file_1_view_0, column Value)
- $w_j$: weight per unit of product $j$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

**Constraints:**
- Shelf capacity for each $i \in S$:
\[
\sum_{j \in P} w_j \, x_{ij} \leq C_i
\]
- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $C_i$: file_0_view_0, columns ShelfID (key), Capacity (value)
- $P$ (products): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, columns ProductName (key), Value (value)
- $w_j$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{ij}$: decision variable for each $(i,j)$ pair with $i$ from file_0_view_0.ShelfID and $j$ from file_1_view_0.ProductName

---

**All parameters, index sets, and variable domains are derived directly from the supplied files and columns.**