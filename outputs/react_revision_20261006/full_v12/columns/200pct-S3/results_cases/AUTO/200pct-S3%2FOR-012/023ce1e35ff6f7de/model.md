### Mathematical Model

**Sets:**
- $I$: set of platforms, indexed by $i$ (from file_0_view_0.resource_id)
- $J$: set of game genres, indexed by $j$ (from file_1_view_0.item_name)

**Parameters:**
- $c_i$: memory capacity of platform $i$ (from file_0_view_0.resource_capacity)
- $v_j$: value per unit of genre $j$ (from file_1_view_0.item_value)
- $a_j$: memory requirement per unit of genre $j$ (from file_1_view_0.resource_requirement)

**Decision Variables:**
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Platform Memory Capacity:**
   \[
   \sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$ (platforms): file_0_view_0.resource_id
- $J$ (game genres): file_1_view_0.item_name
- $c_i$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_j$: file_1_view_0.item_value, keyed by item_name
- $a_j$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)