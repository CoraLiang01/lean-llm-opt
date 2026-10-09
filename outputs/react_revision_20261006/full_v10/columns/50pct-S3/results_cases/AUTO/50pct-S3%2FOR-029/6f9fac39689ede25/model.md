Mathematical Model

Index Sets:
- $I$: set of displays (from file_0_view_0, column ShelfID)
- $J$: set of products (from file_1_view_0, column ProductName)

Parameters:
- $c_i$: capacity of display $i$ (from file_0_view_0, column Capacity, indexed by ShelfID)
- $v_j$: value per unit of product $j$ (from file_1_view_0, column Value, indexed by ProductName)
- $w_j$: weight per unit of product $j$ (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
\]

2. Minimum allocation of the first product (the product in the first row of file_1_view_0, i.e., ProductName = "Smartphone"):
\[
\sum_{i \in I} x_{i, \text{Smartphone}} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Data Mapping

Index Sets:
- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName

Parameters:
- $c_i$: file_0_view_0, column Capacity, indexed by ShelfID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$ (indexed by ShelfID and ProductName)

Special constraint:
- The "first product" is the product in the first row of file_1_view_0 (ProductName = "Smartphone"). The minimum allocation constraint applies to this product.