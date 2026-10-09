#### Index Sets

- $I$: set of platforms (from file_0_view_0, column PlatformId)
- $J$: set of genres (from file_1_view_0, column ProductName)

#### Parameters

- $C_i$: memory capacity of platform $i \in I$ (from file_0_view_0, column Capacity)
- $v_j$: value of one unit of genre $j \in J$ (from file_1_view_0, column Value)
- $w_j$: memory requirement of one unit of genre $j \in J$ (from file_1_view_0, column Weight)

#### Decision Variables

- $x_{ij}$: integer number of units of games from genre $j \in J$ to be listed on platform $i \in I$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

- Platform memory capacity:
  $$
  \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
  $$
- Integer nonnegativity:
  $$
  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
  $$

---

#### Data Mapping

- file_0_view_0 (capacity.csv): PlatformId $\rightarrow$ $I$, Capacity $\rightarrow$ $C_i$
- file_1_view_0 (products.csv): ProductName $\rightarrow$ $J$, Value $\rightarrow$ $v_j$, Weight $\rightarrow$ $w_j$