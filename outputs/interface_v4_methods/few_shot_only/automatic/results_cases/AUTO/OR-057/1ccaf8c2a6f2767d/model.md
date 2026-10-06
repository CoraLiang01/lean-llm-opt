#### Abstract Mathematical Model

**Sets:**
- $I$: Set of platforms, indexed by $i$ (from file_0_view_0, column PlatformID)
- $J$: Set of games, indexed by $j$ (from file_1_view_0, column ProductName)

**Parameters:**
- $c_i$: Memory capacity of platform $i$ (from file_0_view_0, column Capacity)
- $v_j$: Value of game $j$ (from file_1_view_0, column Value)
- $w_j$: Memory requirement of game $j$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of game $j$ to be listed on platform $i$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
- Platform memory capacity:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
- Integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (platforms): file_0_view_0, column PlatformID
- $c_i$: file_0_view_0, column Capacity, keyed by PlatformID
- $J$ (games): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

All data is used as returned, preserving file and row order.