##### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from PlatformID in capacity.csv)
- $J$ = set of game genres, indexed by $j$ (from ProductName in products.csv)

Parameters:
- $c_i$ = capacity of platform $i$ (from Capacity in capacity.csv)
- $v_j$ = value per unit of genre $j$ (from Value in products.csv)
- $w_j$ = memory requirement per unit of genre $j$ (from Weight in products.csv)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of genre $j$ to list on platform $i$

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
- Platform capacity constraints:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

##### Data Mapping

- $I$ (platforms): capacity.csv, column PlatformID
- $c_i$: capacity.csv, column Capacity, keyed by PlatformID
- $J$ (genres): products.csv, column ProductName
- $v_j$: products.csv, column Value, keyed by ProductName
- $w_j$: products.csv, column Weight, keyed by ProductName
- $x_{ij}$: number of units of genre $j$ on platform $i$ (decision variable)