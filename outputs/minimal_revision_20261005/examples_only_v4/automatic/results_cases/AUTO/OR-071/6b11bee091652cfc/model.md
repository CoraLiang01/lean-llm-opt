**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of products (from 41.csv, column "Product Name")
  
**Parameters:**
- $l_p$: Labor required per unit of product $p$ (from 41.csv, column "Labor per unit")
- $m_p$: Material required per unit of product $p$ (from 41.csv, column "Material per unit")
- $s_p$: Selling price per unit of product $p$ (from 41.csv, column "Selling Price")
- $v_p$: Variable cost per unit of product $p$ (from 41.csv, column "Variable Cost")
- $L$: Total available labor per week (given: $1,\!650$)
- $M$: Total available material per week (given: $1,\!850$)
- $F$: Fixed weekly operating cost (given: $4,\!500$)

**Decision Variables:**
- $x_p \geq 0$: Continuous quantity of product $p$ to produce (units)

**Objective:**
\[
\max \left( \sum_{p \in P} (s_p - v_p) x_p - F \right)
\]

**Constraints:**
1. Labor capacity:
   \[
   \sum_{p \in P} l_p x_p \leq L
   \]
2. Material capacity:
   \[
   \sum_{p \in P} m_p x_p \leq M
   \]
3. Nonnegativity:
   \[
   x_p \geq 0 \quad \forall p \in P
   \]

---

**Data Mapping**

- $P$: All records in 41.csv, column "Product Name", table_id: file_0_view_0
- $l_p$: 41.csv, column "Labor per unit", table_id: file_0_view_0, keyed by "Product Name"
- $m_p$: 41.csv, column "Material per unit", table_id: file_0_view_0, keyed by "Product Name"
- $s_p$: 41.csv, column "Selling Price", table_id: file_0_view_0, keyed by "Product Name"
- $v_p$: 41.csv, column "Variable Cost", table_id: file_0_view_0, keyed by "Product Name"
- $L$: $1,\!650$ (from query)
- $M$: $1,\!850$ (from query)
- $F$: $4,\!500$ (from query)
- $x_p$: Decision variable for each $p \in P$ (continuous, $\geq 0$)