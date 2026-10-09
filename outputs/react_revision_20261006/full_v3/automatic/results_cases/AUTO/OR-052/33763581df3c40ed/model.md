Mathematical Model

Sets:
- $I$: set of bookshelves, indexed by $i$ (BookshelfID from file_0_view_0)
- $J$: set of books, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$: capacity of bookshelf $i$ (Capacity from file_0_view_0)
- $v_j$: value of book $j$ (Value from file_1_view_0)
- $w_j$: weight of book $j$ (Weight from file_1_view_0)

Decision Variables:
- $x_{ij}$: number of units of book $j$ to place on bookshelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $I$: BookshelfID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, keyed by BookshelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName