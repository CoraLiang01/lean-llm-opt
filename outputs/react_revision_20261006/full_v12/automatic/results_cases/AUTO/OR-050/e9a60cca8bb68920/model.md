#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with $i \in$ ShelfID from file_0_view_0
- $J$ = set of products (indexed by $j$), with $j \in$ ProductName from file_1_view_0

Parameters:
- $c_i$ = Capacity of display $i$ (from file_0_view_0, column Capacity)
- $v_j$ = Value of product $j$ (from file_1_view_0, column Value)
- $w_j$ = Weight of product $j$ (from file_1_view_0, column Weight)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on display $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
1. Display capacity constraints (for all $i \in I$):
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i
$$

2. Minimum allocation of the first product (for $j^* =$ first ProductName in file_1_view_0, i.e., $j^* = $ "Smartphone"):
$$
\sum_{i \in I} x_{i j^*} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $j^*$: first ProductName in file_1_view_0 (row 0), i.e., "Smartphone"
- $x_{ij}$: number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)