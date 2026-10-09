Mathematical Model

Sets:
- $S$: set of shelves, indexed by $i$ (from file_0_view_0.ShelfID)
- $P$: set of products, indexed by $j$ (from file_1_view_0.ProductName)

Parameters:
- $c_i$: capacity of shelf $i$ (from file_0_view_0.Capacity)
- $v_j$: value of product $j$ (from file_1_view_0.Value)
- $w_j$: weight of product $j$ (from file_1_view_0.Weight)

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in P} w_j x_{ij} \leq c_i \qquad \forall i \in S
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

Data Mapping

- $S$ (shelves): file_0_view_0.ShelfID
- $P$ (products): file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID $i$
- $v_j$: file_1_view_0.Value, keyed by ProductName $j$
- $w_j$: file_1_view_0.Weight, keyed by ProductName $j$
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ (decision variable, indexed by $i \in S$, $j \in P$)