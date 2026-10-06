## Abstract Mathematical Model

**Sets:**
- $I$: Set of platforms, indexed by $i$ (from file_0_view_0.resource_id)
- $J$: Set of game genres, indexed by $j$ (from file_1_view_0.item_name)

**Parameters:**
- $c_i$: Memory capacity of platform $i$ (from file_0_view_0.resource_capacity)
- $v_j$: Value per unit of genre $j$ (from file_1_view_0.item_value)
- $a_j$: Memory requirement per unit of genre $j$ (from file_1_view_0.resource_requirement)

**Decision Variables:**
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

## Data Mapping

- $I$ (platforms): file_0_view_0.resource_id
- $c_i$: file_0_view_0.resource_capacity, keyed by resource_id
- $J$ (genres): file_1_view_0.item_name
- $v_j$: file_1_view_0.item_value, keyed by item_name
- $a_j$: file_1_view_0.resource_requirement, keyed by item_name

All other columns are ignored. No additional constraints are imposed by the data.