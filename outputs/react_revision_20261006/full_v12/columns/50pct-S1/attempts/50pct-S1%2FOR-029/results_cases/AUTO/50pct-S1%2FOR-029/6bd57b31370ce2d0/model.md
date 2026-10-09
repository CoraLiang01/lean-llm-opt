#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with IDs from column ShelfID in file_0_view_0
- $J$ = set of products (indexed by $j$), with names from column ProductName in file_1_view_0
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable), $x_{ij} \in \mathbb{Z}_{\geq 0}$

Parameters:
- $c_i$ = capacity of display $i$ (from Capacity in file_0_view_0)
- $v_j$ = value of product $j$ (from Value in file_1_view_0)
- $w_j$ = weight of product $j$ (from Weight in file_1_view_0)

Let $j^*$ denote the index of the first product in file_1_view_0 (ProductName = "Smartphone").

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i
\]
2. Minimum allocation of the first product:
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $j^*$: ProductName = "Smartphone" (first row of file_1_view_0)
- $x_{ij}$: Number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)