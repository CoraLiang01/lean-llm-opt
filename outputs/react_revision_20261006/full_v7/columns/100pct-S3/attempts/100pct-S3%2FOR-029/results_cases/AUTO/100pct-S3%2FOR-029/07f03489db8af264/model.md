#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business IDs from file_0_view_0.ShelfID
- $J$ = set of products (indexed by $j$), with business IDs from file_1_view_0.ProductName

Parameters:
- $c_i$ = capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID)
- $v_j$ = value per unit of product $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$ = weight per unit of product $j$ (file_1_view_0, column Weight, key ProductName)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (let $j^*$ be the ProductName in the first row of file_1_view_0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: file_0_view_0.ShelfID (business ID for displays)
- $J$: file_1_view_0.ProductName (business ID for products)
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $j^*$: ProductName in file_1_view_0, source_row 0 ("Smartphone")
- $x_{ij}$: number of units of product $j$ placed on display $i$ (indexed by ShelfID and ProductName)

All index sets, parameters, and constraints are mapped directly from the supplied CSV data as described above.