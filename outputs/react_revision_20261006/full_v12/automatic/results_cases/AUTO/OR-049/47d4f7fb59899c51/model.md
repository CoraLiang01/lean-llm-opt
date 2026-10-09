### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value of product $j$ (from Value in file_1_view_0)
- $w_j$ = weight of product $j$ (from Weight in file_1_view_0)
- $C_i$ = capacity of shelf $i$ (from Capacity in file_0_view_0)

#### Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

#### Constraints:
- Shelf capacity for each shelf $i$:
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]
- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

### Data Mapping

- $I$: All ShelfID from file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $C_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Decision variable for each $(i,j)$ pair

All indices, parameters, and constraints are mapped directly to the original CSV columns and business identifiers.