**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products (from 41.csv, column "Product Name").

**Parameters:**
- $l_i$: Labor required per unit of product $i$ (from 41.csv, column "Labor per unit").
- $m_i$: Material required per unit of product $i$ (from 41.csv, column "Material per unit").
- $s_i$: Selling price per unit of product $i$ (from 41.csv, column "Selling Price").
- $v_i$: Variable cost per unit of product $i$ (from 41.csv, column "Variable Cost").
- $L$: Total available labor per week ($L = 1650$).
- $M$: Total available material per week ($M = 1850$).
- $F$: Fixed weekly operating cost ($F = 4500$).

**Decision Variables:**
- $x_i \geq 0$: Production quantity of product $i$ (continuous).

**Objective:**
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

**Constraints:**
1. Labor capacity:
   \[
   \sum_{i \in I} l_i x_i \leq L
   \]
2. Material capacity:
   \[
   \sum_{i \in I} m_i x_i \leq M
   \]
3. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: file_0_view_0, column "Product Name"
- $l_i$: file_0_view_0, column "Labor per unit"
- $m_i$: file_0_view_0, column "Material per unit"
- $s_i$: file_0_view_0, column "Selling Price"
- $v_i$: file_0_view_0, column "Variable Cost"
- $L = 1650$, $M = 1850$, $F = 4500$: (from user query)