ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of Bookshelf IDs (from file_0_view_0, column BookshelfID)
- $J$: set of Book Product Names (from file_1_view_0, column ProductName)

Parameters:
- $c_i$: capacity of bookshelf $i$ (from file_0_view_0, column Capacity, indexed by BookshelfID)
- $v_j$: value of book $j$ (from file_1_view_0, column Value, indexed by ProductName)
- $w_j$: weight of book $j$ (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- $x_{ij}$: number of units of book $j$ to place on bookshelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

DATA MAPPING

- $I$: BookshelfID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, keyed by BookshelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: integer variable for each $(i,j) \in I \times J$ (bookshelf, book) pair

All parameters and index sets are defined directly from the returned data; no values are hard-coded.