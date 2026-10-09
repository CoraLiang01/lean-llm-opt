Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

Define:
- $S$ = set of StorageIDs: $\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $P$ = set of ProductNames: 
  - Window Unit
  - Portable Unit
  - Split System
  - Ductless System
  - Central AC
  - Hybrid AC
  - Geothermal AC
  - Smart AC
  - Evaporative Cooler
  - Package Unit

Let:
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of storage area $i$ (from capacity.csv)

The model is:

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

Where the parameters are:

Storage Areas and Capacities (from capacity.csv, in source order):

| StorageID | Capacity |
|-----------|----------|
| 1         | 1083     |
| 2         | 1840     |
| 3         | 770      |
| 4         | 1299     |
| 5         | 1259     |
| 6         | 543      |
| 7         | 1831     |
| 8         | 855      |
| 9         | 619      |
| 10        | 637      |
| 11        | 935      |
| 12        | 626      |
| 13        | 1457     |
| 14        | 1198     |
| 15        | 837      |

Product Types, Values, and Weights (from products.csv, in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Window Unit         | 4811  | 114    |
| Portable Unit       | 1130  | 200    |
| Split System        | 1611  | 106    |
| Ductless System     | 3368  | 256    |
| Central AC          | 2135  | 268    |
| Hybrid AC           | 1046  | 185    |
| Geothermal AC       | 4030  | 299    |
| Smart AC            | 3761  | 131    |
| Evaporative Cooler  | 3523  | 139    |
| Package Unit        | 1701  | 105    |

Decision variables:
\[
x_{ij} = \text{number of units of product } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Capacity constraints (for each $i \in S$):
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]