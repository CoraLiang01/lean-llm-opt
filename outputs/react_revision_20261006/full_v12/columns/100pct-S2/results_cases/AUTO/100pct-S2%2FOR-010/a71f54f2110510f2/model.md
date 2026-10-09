### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (from file_0_view_0, column SectionID)
- $J$ = set of products, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $c_i$ = capacity of section $i$ (file_0_view_0, column Capacity)
- $v_j$ = value (revenue) per unit of product $j$ (file_1_view_0, column Value)
- $w_j$ = space requirement per unit of product $j$ (file_1_view_0, column Weight)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

Subject to:
$$
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$: All SectionID in file_0_view_0 (capacity.csv), column SectionID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0 (capacity.csv), column Capacity, keyed by SectionID
- $v_j$: file_1_view_0 (products.csv), column Value, keyed by ProductName
- $w_j$: file_1_view_0 (products.csv), column Weight, keyed by ProductName
- $x_{ij}$: Decision variable for each $(i,j)$ pair

All parameters and index sets are defined directly from the current CSV data as described above.