#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of sections (indexed by $i$), from `capacity.csv` column `SectionID`
- $P$: set of products (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: display space capacity of section $i$, from `capacity.csv` column `Capacity`
- $v_j$: price (revenue per unit) of product $j$, from `products.csv` column `Value`
- $w_j$: shelf space required per unit of product $j$, from `products.csv` column `Weight$

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ to stock in section $i$, integer, $x_{ij} \geq 0$

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Section Capacity Constraints:**
   \[
   \sum_{j \in P} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in S
   \]
2. **Integrality Constraints:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   \]

---

**Data Mapping:**

- `capacity.csv`:
  - Section index set $S$ from column `SectionID`
  - Section capacity $C_i$ from column `Capacity`
- `products.csv`:
  - Product index set $P$ from column `ProductName`
  - Product price $v_j$ from column `Value`
  - Product shelf space $w_j$ from column `Weight`