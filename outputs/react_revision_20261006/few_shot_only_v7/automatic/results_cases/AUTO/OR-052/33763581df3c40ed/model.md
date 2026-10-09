Mathematical Optimization Model

Sets:
- $S$: set of bookshelves, indexed by $i$ (from capacity.csv, column BookshelfID)
- $B$: set of books, indexed by $j$ (from products.csv, column ProductName)

Parameters:
- $c_i$: capacity of bookshelf $i$ (from capacity.csv, column Capacity)
- $v_j$: value of book $j$ (from products.csv, column Value)
- $w_j$: weight of book $j$ (from products.csv, column Weight)

Decision Variables:
- $x_{ij}$: number of units of book $j$ placed on bookshelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in S} \sum_{j \in B} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in B} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in B
\]

Data Mapping

- $S$ (bookshelves): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv, column BookshelfID
- $B$ (books): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv, column ProductName
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv, column Capacity
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv, column Value
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv, column Weight