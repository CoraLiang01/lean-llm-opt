#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to `ShelfID` in `file_0_view_0` (capacity.csv)
- $J$: set of products (indexed by $j$), corresponding to `ProductName` in `file_1_view_0` (products.csv)

**Parameters:**
- $c_i$: capacity of display $i$ (`Capacity` in `file_0_view_0`)
- $v_j$: value of product $j$ (`Value` in `file_1_view_0`)
- $w_j$: weight of product $j$ (`Weight` in `file_1_view_0$)
- $j^*$: index of the first product in source order (row 0 of `file_1_view_0`)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i
   \]
2. **Minimum Allocation of First Product:**  
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (Displays): `file_0_view_0.ShelfID`
- $c_i$: `file_0_view_0.Capacity` (for each $i$)
- $J$ (Products): `file_1_view_0.ProductName`
- $v_j$: `file_1_view_0.Value` (for each $j$)
- $w_j$: `file_1_view_0.Weight` (for each $j$)
- $j^*$: product with `file_1_view_0.source_row = 0` (the first product in products.csv)

All parameters and indices are to be used exactly as returned, preserving source order and identifiers.