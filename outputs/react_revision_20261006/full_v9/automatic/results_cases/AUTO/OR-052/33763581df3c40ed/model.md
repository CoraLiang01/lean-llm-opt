##### Mathematical Model

Let:
- $I$ = set of BookshelfIDs from file_0_view_0 (capacity.csv)
- $J$ = set of ProductNames from file_1_view_0 (products.csv)
- $c_i$ = Capacity of bookshelf $i \in I$ (from file_0_view_0, column Capacity)
- $v_j$ = Value of book $j \in J$ (from file_1_view_0, column Value)
- $w_j$ = Weight of book $j \in J$ (from file_1_view_0, column Weight)
- $x_{ij}$ = number of units of book $j$ placed on bookshelf $i$ (decision variable, integer, $\geq 0$)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$: BookshelfID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, keyed by BookshelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: integer variable for each $(i, j) \in I \times J$ (number of units of book $j$ on bookshelf $i$)