ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of ShelfIDs (displays), from file_0_view_0, column ShelfID
- $J$: set of ProductNames (products), from file_1_view_0, column ProductName

Parameters:
- $c_i$: capacity of display $i$, from file_0_view_0, column Capacity
- $v_j$: value of product $j$, from file_1_view_0, column Value
- $w_j$: weight of product $j$, from file_1_view_0, column Weight

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (for $j^* =$ first ProductName in file_1_view_0, i.e., source_row 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

DATA MAPPING

- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $j^*$: ProductName from file_1_view_0, source_row 0

All variables, parameters, and constraints are indexed and mapped as above. No data is omitted or synthesized.