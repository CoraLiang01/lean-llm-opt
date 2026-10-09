ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of shelves (indexed by $i$), from file_0_view_0.ShelfID
- $J$: set of products (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: capacity of shelf $i$, from file_0_view_0.Capacity
- $v_j$: value per unit of product $j$, from file_1_view_0.Value
- $w_j$: weight (space requirement) per unit of product $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

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

- $I$: file_0_view_0.ShelfID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID $i$
- $v_j$: file_1_view_0.Value, keyed by ProductName $j$
- $w_j$: file_1_view_0.Weight, keyed by ProductName $j$
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ (decision variable)