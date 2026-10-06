**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of platforms, indexed by $i$. (from `file_0_view_0`, column `PlatformId`)
- $J$: Set of game genres, indexed by $j$. (from `file_1_view_0`, column `ProductName`)

**Parameters:**
- $c_i$: Memory capacity of platform $i$. (from `file_0_view_0`, column `Capacity`)
- $v_j$: Value per unit of genre $j$. (from `file_1_view_0`, column `Value`)
- $w_j$: Memory requirement per unit of genre $j$. (from `file_1_view_0`, column `Weight`)

**Decision Variables:**
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Platform Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i, \quad \forall i \in I
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (platforms): All records in `file_0_view_0`, column `PlatformId`
- $J$ (genres): All records in `file_1_view_0`, column `ProductName`
- $c_i$: `file_0_view_0`, columns `PlatformId`, `Capacity`
- $v_j$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_j$: `file_1_view_0`, columns `ProductName`, `Weight`