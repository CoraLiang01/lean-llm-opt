Let $x_i$ be the production quantity (in units) of product $i$, where $i$ indexes the 198 products as listed in the "Product Name" column of 41.csv.

**Parameters (from 41.csv, for each product $i$):**
- $l_i$: Labor per unit (from "Labor per unit")
- $m_i$: Material per unit (from "Material per unit")
- $s_i$: Selling Price (from "Selling Price")
- $v_i$: Variable Cost (from "Variable Cost")

**Constants:**
- Total labor available: $1650$
- Total material available: $1850$
- Fixed weekly operating cost: $4500$

**Model:**

Maximize
$$
\sum_{i=1}^{198} (s_i - v_i) x_i - 4500
$$

Subject to
$$
\sum_{i=1}^{198} l_i x_i \leq 1650
$$

$$
\sum_{i=1}^{198} m_i x_i \leq 1850
$$

$$
x_i \geq 0 \quad \forall i \in \{1,2,\ldots,198\}
$$

Where for each $i$, the coefficients $l_i$, $m_i$, $s_i$, $v_i$ are taken directly from the corresponding columns and rows of 41.csv, preserving the original order. Each $x_i$ is a nonnegative continuous variable representing the weekly production quantity of product $i$.