Mathematical Model

Index Sets:
Let $I$ be the set of BookshelfIDs from file_0_view_0 (capacity.csv).
Let $J$ be the set of ProductNames from file_1_view_0 (products.csv).

Parameters:
For each $i \in I$, let $C_i$ be the Capacity from file_0_view_0, column Capacity.
For each $j \in J$, let $v_j$ be the Value from file_1_view_0, column Value.
For each $j \in J$, let $w_j$ be the Weight from file_1_view_0, column Weight.

Decision Variables:
For each $i \in I$, $j \in J$:
$x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of book $j$ placed on bookshelf $i$.

Objective:
$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

Subject to:
For each $i \in I$:
$\sum_{j \in J} w_j x_{ij} \leq C_i$

For all $i \in I$, $j \in J$:
$x_{ij} \in \mathbb{Z}_{\geq 0}$

Data Mapping

Index Sets:
$I$ = BookshelfID from file_0_view_0 (capacity.csv)
$J$ = ProductName from file_1_view_0 (products.csv)

Parameters:
$C_i$ = file_0_view_0, column Capacity, keyed by BookshelfID
$v_j$ = file_1_view_0, column Value, keyed by ProductName
$w_j$ = file_1_view_0, column Weight, keyed by ProductName

Decision Variables:
$x_{ij}$: number of units of ProductName $j$ placed on BookshelfID $i$ (integer, $\geq 0$)