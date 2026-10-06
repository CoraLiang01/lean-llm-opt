**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of platforms, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$: Set of game genres, indexed by $j$ (from all item_name in file_1_view_0)

**Parameters:**
- $c_i$: Memory capacity of platform $i$ (resource_capacity from file_0_view_0, indexed by resource_id)
- $v_j$: Value per unit of genre $j$ (item_value from file_1_view_0, indexed by item_name)
- $a_j$: Memory requirement per unit of genre $j$ (resource_requirement from file_1_view_0, indexed by item_name)

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
2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (platforms): All resource_id in file_0_view_0 (capacity.csv)
- $J$ (genres): All item_name in file_1_view_0 (products.csv)
- $c_i$: resource_capacity in file_0_view_0, indexed by resource_id
- $v_j$: item_value in file_1_view_0, indexed by item_name
- $a_j$: resource_requirement in file_1_view_0, indexed by item_name

**Variable Domain:** $x_{ij}$ are nonnegative integers for all $i \in I$, $j \in J$.