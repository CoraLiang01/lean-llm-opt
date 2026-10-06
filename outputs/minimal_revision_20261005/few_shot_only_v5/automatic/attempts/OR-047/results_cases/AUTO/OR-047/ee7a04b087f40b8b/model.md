**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of platforms (indexed by $i$), from capacity.csv [PlatformId]
- $J$: Set of game genres (indexed by $j$), from products.csv [ProductName]

**Parameters:**
- $c_i$: Memory capacity of platform $i$ (from capacity.csv [Capacity])
- $v_j$: Value per unit of genre $j$ (from products.csv [Value])
- $w_j$: Memory requirement per unit of genre $j$ (from products.csv [Weight])

**Decision Variables:**
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

- **Platform Memory Capacity Constraints:**
  \[
  \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
  \]

- **Integrality and Nonnegativity:**
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
  \]

---

**Data Mapping**

- $I$ (platforms): capacity.csv [PlatformId]
- $c_i$: capacity.csv [Capacity], keyed by [PlatformId]
- $J$ (genres): products.csv [ProductName]
- $v_j$: products.csv [Value], keyed by [ProductName]
- $w_j$: products.csv [Weight], keyed by [ProductName]
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)