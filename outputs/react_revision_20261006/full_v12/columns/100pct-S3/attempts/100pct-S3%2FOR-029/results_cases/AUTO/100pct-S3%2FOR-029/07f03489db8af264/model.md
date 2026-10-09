#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with IDs from file_0_view_0.ShelfID
- $J$ = set of products (indexed by $j$), with names from file_1_view_0.ProductName

Parameters:
- $c_i$ = capacity of display $i$ (file_0_view_0.Capacity, for ShelfID $i$)
- $v_j$ = value per unit of product $j$ (file_1_view_0.Value, for ProductName $j$)
- $w_j$ = weight per unit of product $j$ (file_1_view_0.Weight, for ProductName $j$)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$ (displays): file_0_view_0.ShelfID
- $J$ (products): file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity (for ShelfID $i$)
- $v_j$: file_1_view_0.Value (for ProductName $j$)
- $w_j$: file_1_view_0.Weight (for ProductName $j$)
- $j^*$: ProductName in file_1_view_0, source_row = 0

All indices, parameters, and constraints are mapped directly to the corresponding columns and rows in the provided CSV files.