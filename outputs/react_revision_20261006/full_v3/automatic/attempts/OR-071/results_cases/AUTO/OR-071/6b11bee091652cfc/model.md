##### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with business identifier "Product Name" from 41.csv).

**Parameters:**
- $l_i$: labor required per unit of product $i$ ("Labor per unit", 41.csv)
- $m_i$: material required per unit of product $i$ ("Material per unit", 41.csv)
- $s_i$: selling price per unit of product $i$ ("Selling Price", 41.csv)
- $v_i$: variable cost per unit of product $i$ ("Variable Cost", 41.csv)
- $L = 1650$: total weekly labor capacity
- $M = 1850$: total weekly material capacity
- $F = 4500$: fixed weekly operating cost

**Decision Variables:**
- $x_i \geq 0$: production quantity of product $i$ (continuous)

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

##### Data Mapping

- $I$: All rows in table_id file_0_view_0, column "Product Name"
- $l_i$: file_0_view_0, column "Labor per unit", keyed by "Product Name"
- $m_i$: file_0_view_0, column "Material per unit", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Selling Price", keyed by "Product Name"
- $v_i$: file_0_view_0, column "Variable Cost", keyed by "Product Name"
- $L = 1650$, $M = 1850$, $F = 4500$: from user description

- Decision variable $x_i$ is the nonnegative continuous production quantity for each $i \in I$ ("Product Name", file_0_view_0).