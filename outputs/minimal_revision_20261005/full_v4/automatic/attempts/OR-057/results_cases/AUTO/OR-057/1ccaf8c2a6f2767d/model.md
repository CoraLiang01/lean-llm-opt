**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of platforms, indexed by $i$, with business identifier PlatformID from file_0_view_0.
- $J$: Set of games, indexed by $j$, with business identifier ProductName from file_1_view_0.

**Parameters:**
- $c_i$: Capacity of platform $i$ (from file_0_view_0, column Capacity).
- $v_j$: Value of game $j$ (from file_1_view_0, column Value).
- $w_j$: Memory requirement (weight) of game $j$ (from file_1_view_0, column Weight).

**Decision Variables:**
- $x_{ij}$: Number of units of game $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Platform Capacity Constraints:**  
   For each platform $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (platforms): file_0_view_0, column PlatformID
- $J$ (games): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by PlatformID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

**Variables:**
- $x_{ij}$: Number of units of game $j$ on platform $i$ (integer, $\geq 0$)