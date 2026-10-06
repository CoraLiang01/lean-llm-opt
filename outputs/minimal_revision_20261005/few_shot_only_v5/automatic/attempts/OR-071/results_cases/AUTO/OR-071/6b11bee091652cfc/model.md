**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products (from 41.csv, column "Product Name")

**Parameters:**
- $l_i$: Labor required per unit of product $i$ (from "Labor per unit")
- $m_i$: Material required per unit of product $i$ (from "Material per unit")
- $s_i$: Selling price per unit of product $i$ (from "Selling Price")
- $v_i$: Variable cost per unit of product $i$ (from "Variable Cost")
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

| Model Symbol | Source Table/Column                                 | Description                                 |
|--------------|-----------------------------------------------------|---------------------------------------------|
| $I$          | 41.csv, "Product Name"                              | Set of all products                         |
| $l_i$        | 41.csv, "Labor per unit"                            | Labor required per unit of product $i$      |
| $m_i$        | 41.csv, "Material per unit"                         | Material required per unit of product $i$   |
| $s_i$        | 41.csv, "Selling Price"                             | Selling price per unit of product $i$       |
| $v_i$        | 41.csv, "Variable Cost"                             | Variable cost per unit of product $i$       |
| $L$          | (given in query)                                    | $1650$ (weekly labor capacity)              |
| $M$          | (given in query)                                    | $1850$ (weekly material capacity)           |
| $F$          | (given in query)                                    | $4500$ (fixed weekly operating cost)        |
| $x_i$        | (decision variable, continuous, $\geq 0$)           | Production quantity of product $i$          |