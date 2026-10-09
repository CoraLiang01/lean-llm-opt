### Mathematical Model

**Sets:**
- $I$: set of platforms, indexed by $i$ (from `file_0_view_0.resource_id`)
- $J$: set of game genres, indexed by $j$ (from `file_1_view_0.item_name`)

**Parameters:**
- $c_i$: memory capacity of platform $i$ (`file_0_view_0.resource_capacity`)
- $v_j$: value per unit of genre $j$ (`file_1_view_0.item_value`)
- $a_j$: memory requirement per unit of genre $j$ (`file_1_view_0.resource_requirement`)

**Decision Variables:**
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
- Platform memory capacity:
\[
\sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
- Integer and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

### Data Mapping

- $I$: All `resource_id` in `file_0_view_0` (capacity.csv)
- $J$: All `item_name` in `file_1_view_0` (products.csv)
- $c_i$: `resource_capacity` from `file_0_view_0` (capacity.csv), keyed by `resource_id`
- $v_j$: `item_value` from `file_1_view_0` (products.csv), keyed by `item_name`
- $a_j$: `resource_requirement` from `file_1_view_0` (products.csv), keyed by `item_name`
- $x_{ij}$: Number of units of genre $j$ on platform $i$ (decision variable, integer, $\geq 0$)