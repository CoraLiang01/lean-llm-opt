#### Abstract Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business key ShelfID from file_0_view_0.
- $J$ = set of products (indexed by $j$), with business key ProductName from file_1_view_0.

Parameters:
- $c_i$ = capacity of display $i$ (from Capacity in file_0_view_0, indexed by ShelfID).
- $v_j$ = value per unit of product $j$ (from Value in file_1_view_0, indexed by ProductName).
- $w_j$ = weight per unit of product $j$ (from Weight in file_1_view_0, indexed by ProductName).

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation for the first product (let $j^*$ be the ProductName in file_1_view_0 with source_row = 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (displays): ShelfID from file_0_view_0 (capacity.csv)
- $J$ (products): ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column in file_0_view_0, indexed by ShelfID
- $v_j$: Value column in file_1_view_0, indexed by ProductName
- $w_j$: Weight column in file_1_view_0, indexed by ProductName
- $j^*$: ProductName in file_1_view_0, source_row = 0 (first product as ordered in products.csv)

All indices, parameters, and constraints are mapped directly to the original file columns and row order as required.