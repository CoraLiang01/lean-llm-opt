#### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$, with platform IDs from capacity.csv ("resource_id")
- $J$ = set of game genres, indexed by $j$, with genre names from products.csv ("item_name")
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer $\geq 0$)
- $v_j$ = value of one unit of genre $j$ ("item_value")
- $a_j$ = memory requirement of one unit of genre $j$ ("resource_requirement")
- $c_i$ = memory capacity of platform $i$ ("resource_capacity")

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

- Platform memory capacity constraints:
\[
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (platforms): file_0_view_0, column "resource_id"
- $c_i$: file_0_view_0, column "resource_capacity", keyed by "resource_id"
- $J$ (genres): file_1_view_0, column "item_name"
- $v_j$: file_1_view_0, column "item_value", keyed by "item_name"
- $a_j$: file_1_view_0, column "resource_requirement", keyed by "item_name"
- $x_{ij}$: decision variable for platform $i$ and genre $j$ (integer, $\geq 0$)