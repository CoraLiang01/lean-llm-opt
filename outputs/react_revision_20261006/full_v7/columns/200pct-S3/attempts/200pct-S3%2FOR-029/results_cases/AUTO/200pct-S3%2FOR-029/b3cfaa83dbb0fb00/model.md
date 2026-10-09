##### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$ = set of products (indexed by $j$), corresponding to all ProductName in file_1_view_1.
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable), $x_{ij} \in \mathbb{Z}_{\geq 0}$.
- $v_j$ = value of product $j$ (from Value in file_1_view_1).
- $w_j$ = weight of product $j$ (from Weight in file_1_view_1).
- $C_i$ = capacity of display $i$ (from Capacity in file_0_view_0).
- $j^*$ = index of the first product in file_1_view_1 (i.e., the product with the smallest source_row, which is "Smartphone").

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
\]
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column "ShelfID".
- $C_i$: file_0_view_0, column "Capacity", mapped by ShelfID.
- $J$: All ProductName in file_1_view_1 (products.csv), column "ProductName".
- $v_j$: file_1_view_1, column "Value", mapped by ProductName.
- $w_j$: file_1_view_1, column "Weight", mapped by ProductName.
- $j^*$: ProductName in file_1_view_1 with the smallest source_row ("Smartphone").
- $x_{ij}$: Number of units of product $j$ placed on display $i$, integer, nonnegative.