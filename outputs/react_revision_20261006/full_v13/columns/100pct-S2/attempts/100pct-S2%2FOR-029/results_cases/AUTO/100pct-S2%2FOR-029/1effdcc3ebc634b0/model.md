#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business IDs from file_0_view_0.ShelfID
- $J$ = set of products (indexed by $j$), with business IDs from file_1_view_0.ProductName

Parameters:
- $c_i$ = capacity of display $i$ (file_0_view_0.Capacity)
- $v_j$ = value per unit of product $j$ (file_1_view_0.Value)
- $w_j$ = weight per unit of product $j$ (file_1_view_0.Weight)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
1. **Display capacity constraints** (for all $i \in I$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i
\]
2. **Minimum allocation of first product** (let $j^*$ be the business ID of the first product in file_1_view_0, i.e., ProductName in source_row 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. **Nonnegativity and integrality**:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: file_0_view_0.ShelfID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $j^*$: file_1_view_0.ProductName where source_row = 0 (i.e., the first product in products.csv)
- $x_{ij}$: number of units of product $j$ placed on display $i$ (decision variable, indexed by $I \times J$)