Mathematical Model

Index Sets:
- $I$: set of BookshelfIDs (from file_0_view_0, column "BookshelfID")
- $J$: set of ProductNames (from file_1_view_0, column "ProductName")

Parameters:
- $c_i$: Capacity of bookshelf $i$ (from file_0_view_0, column "Capacity")
- $v_j$: Value of book $j$ (from file_1_view_0, column "Value")
- $w_j$: Weight of book $j$ (from file_1_view_0, column "Weight")

Decision Variables:
- $x_{ij}$: number of units of book $j$ to place on bookshelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $I$: file_0_view_0, column "BookshelfID"
- $J$: file_1_view_0, column "ProductName"
- $c_i$: file_0_view_0, column "Capacity", keyed by "BookshelfID"
- $v_j$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_j$: file_1_view_0, column "Weight", keyed by "ProductName"
- $x_{ij}$: allocation variable for bookshelf $i$ and book $j$