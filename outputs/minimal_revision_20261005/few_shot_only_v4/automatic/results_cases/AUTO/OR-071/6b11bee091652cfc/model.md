**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products (from column "Product Name" in 41.csv)

**Parameters:**
- $l_i$: Labor required per unit of product $i$ (from "Labor per unit", 41.csv)
- $m_i$: Material required per unit of product $i$ (from "Material per unit", 41.csv)
- $s_i$: Selling price per unit of product $i$ (from "Selling Price", 41.csv)
- $v_i$: Variable cost per unit of product $i$ (from "Variable Cost", 41.csv)
- $L$: Total available labor per week ($L = 1650$)
- $M$: Total available material per week ($M = 1850$)
- $F$: Fixed weekly operating cost ($F = 4500$)

**Decision Variables:**
- $x_i \geq 0$: Continuous quantity of product $i$ to produce (units)

**Objective:**
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

**Constraints:**
1. **Labor Capacity:**
   \[
   \sum_{i \in I} l_i x_i \leq L
   \]
2. **Material Capacity:**
   \[
   \sum_{i \in I} m_i x_i \leq M
   \]
3. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

---

**Data Mapping**

| Symbol         | Source Table/Column                                 |
|----------------|----------------------------------------------------|
| $I$            | 41.csv, "Product Name"                             |
| $l_i$          | 41.csv, "Labor per unit"                           |
| $m_i$          | 41.csv, "Material per unit"                        |
| $s_i$          | 41.csv, "Selling Price"                            |
| $v_i$          | 41.csv, "Variable Cost"                            |
| $L$            | (Given in query)                                   |
| $M$            | (Given in query)                                   |
| $F$            | (Given in query)                                   |
| $x_i$          | Decision variable for each $i \in I$               |