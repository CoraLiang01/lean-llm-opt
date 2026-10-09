##### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with elements given by the "ShelfID" column in file_0_view_0.
- $J$ = set of products (indexed by $j$), with elements given by the "ProductName" column in file_1_view_0.

Parameters:
- $c_i$ = capacity of display $i$ (from "Capacity" in file_0_view_0).
- $v_j$ = value of product $j$ (from "Value" in file_1_view_0).
- $w_j$ = weight of product $j$ (from "Weight" in file_1_view_0).

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
2. Minimum allocation of the first product (let $j^*$ be the product with the smallest source_row in file_1_view_0, i.e., the first product listed):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: All "ShelfID" values from file_0_view_0 (capacity.csv), in source order.
- $J$: All "ProductName" values from file_1_view_0 (products.csv), in source order.
- $c_i$: "Capacity" column in file_0_view_0, mapped by "ShelfID".
- $v_j$: "Value" column in file_1_view_0, mapped by "ProductName".
- $w_j$: "Weight" column in file_1_view_0, mapped by "ProductName".
- $j^*$: The "ProductName" in file_1_view_0 with source_row = 0 (i.e., the first product listed).