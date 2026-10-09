### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$ = set of game genres, indexed by $j$ (from all item_name in file_1_view_0)
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (integer, $\geq 0$)

Parameters:
- $v_j$ = item_value of genre $j$ (from file_1_view_0)
- $a_j$ = resource_requirement of genre $j$ (from file_1_view_0)
- $c_i$ = resource_capacity of platform $i$ (from file_0_view_0)

#### Objective:
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

#### Constraints:
- Platform memory capacity:
$$
\sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$: All platform resource_id from file_0_view_0 (capacity.csv)
- $J$: All item_name from file_1_view_0 (products.csv)
- $v_j$: file_1_view_0, column item_value, key item_name
- $a_j$: file_1_view_0, column resource_requirement, key item_name
- $c_i$: file_0_view_0, column resource_capacity, key resource_id
- $x_{ij}$: integer decision variable for each $(i,j)$

All indices, parameters, and constraints are mapped directly to the columns and keys as specified above.