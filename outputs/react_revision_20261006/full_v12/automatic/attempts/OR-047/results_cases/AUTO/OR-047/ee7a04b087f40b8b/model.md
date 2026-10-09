### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from PlatformId in file_0_view_0)
- $J$ = set of game genres, indexed by $j$ (from ProductName in file_1_view_0)
- $c_i$ = memory capacity of platform $i$ (Capacity in file_0_view_0)
- $v_j$ = value per unit of genre $j$ (Value in file_1_view_0)
- $w_j$ = memory requirement per unit of genre $j$ (Weight in file_1_view_0)
- $x_{ij}$ = number of units of genre $j$ to list on platform $i$ (decision variable, integer, $\geq 0$)

#### Objective:
Maximize total value across all platforms:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

#### Constraints:
- Platform memory capacity:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$: PlatformId from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, column "Capacity", key PlatformId
- $v_j$: Value from file_1_view_0, column "Value", key ProductName
- $w_j$: Weight from file_1_view_0, column "Weight", key ProductName
- $x_{ij}$: Number of units of genre $j$ on platform $i$ (decision variable, integer, $\geq 0$)