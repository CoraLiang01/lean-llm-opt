#### Abstract Mathematical Model

**Index Sets:**
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
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Platform Memory Capacity:**
   \[
   \sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (platforms): file_0_view_0.resource_id
- $J$ (genres): file_1_view_0.item_name
- $c_i$: file_0_view_0.resource_capacity, indexed by resource_id
- $v_j$: file_1_view_0.item_value, indexed by item_name
- $a_j$: file_1_view_0.resource_requirement, indexed by item_name

Each $x_{ij}$ is the number of units of games from genre $j$ to be listed on platform $i$. All parameters and index sets are mapped directly from the validated source columns as above.