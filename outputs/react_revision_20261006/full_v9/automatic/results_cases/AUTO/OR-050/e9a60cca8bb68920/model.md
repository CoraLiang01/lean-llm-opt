Mathematical Model

Sets:
- $I$: set of ShelfIDs (displays), from file_0_view_0, column ShelfID
- $J$: set of ProductNames (products), from file_1_view_0, column ProductName

Parameters:
- $c_i$: capacity of display $i \in I$ (file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value of product $j \in J$ (file_1_view_0, column Value, key ProductName)
- $w_j$: weight of product $j \in J$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
\]

2. Minimum allocation of the first product (the product in the first row of file_1_view_0, column ProductName, denoted $j^*$):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $j^*$: first row of file_1_view_0, column ProductName ("Smartphone")
- $x_{ij}$: number of units of product $j$ placed on display $i$ (decision variable, indexed by $I \times J$)