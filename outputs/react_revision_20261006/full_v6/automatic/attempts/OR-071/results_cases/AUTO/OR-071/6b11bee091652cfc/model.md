## Mathematical Model

Let $I$ be the set of products, indexed by $i$ (corresponding to "Product Name" in 41.csv).

**Parameters:**
- $l_i$: Labor required per unit of product $i$ ("Labor per unit", 41.csv)
- $m_i$: Material required per unit of product $i$ ("Material per unit", 41.csv)
- $s_i$: Selling price per unit of product $i$ ("Selling Price", 41.csv)
- $v_i$: Variable cost per unit of product $i$ ("Variable Cost", 41.csv)
- $L = 1650$: Total weekly labor capacity
- $M = 1850$: Total weekly material capacity
- $F = 4500$: Fixed weekly operating cost

**Decision Variables:**
- $x_i \geq 0$: Continuous quantity of product $i$ to produce (units)

**Objective:**
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

**Constraints:**
\[
\sum_{i \in I} l_i x_i \leq L
\]
\[
\sum_{i \in I} m_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

---

## Data Mapping

- $I$: All rows in table_id = file_0_view_0, column "Product Name"
- $l_i$: file_0_view_0, column "Labor per unit", keyed by "Product Name"
- $m_i$: file_0_view_0, column "Material per unit", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Selling Price", keyed by "Product Name"
- $v_i$: file_0_view_0, column "Variable Cost", keyed by "Product Name"
- $L = 1650$, $M = 1850$, $F = 4500$: from user description

**Variables:**
- $x_i$: continuous, nonnegative, for each $i \in I$ ("Product Name", file_0_view_0)