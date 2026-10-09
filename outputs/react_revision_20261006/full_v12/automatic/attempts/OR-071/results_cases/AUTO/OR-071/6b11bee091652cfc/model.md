### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (from all "Product Name" in 41.csv).

**Parameters:**
- $l_i$: Labor per unit of product $i$ ("Labor per unit", 41.csv)
- $m_i$: Material per unit of product $i$ ("Material per unit", 41.csv)
- $p_i$: Selling price per unit of product $i$ ("Selling Price", 41.csv)
- $v_i$: Variable cost per unit of product $i$ ("Variable Cost", 41.csv)
- $L$: Total weekly labor capacity $=1650$
- $M$: Total weekly material capacity $=1850$
- $F$: Fixed weekly operating cost $=4500$

**Decision Variables:**
- $x_i \geq 0$: Production quantity of product $i$ (continuous)

**Objective:**
\[
\max \left( \sum_{i \in I} (p_i - v_i) x_i - F \right)
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

### Data Mapping

- $I$: All products from 41.csv, column "Product Name", table_id: file_0_view_0
- $l_i$: "Labor per unit", table_id: file_0_view_0, key: "Product Name"
- $m_i$: "Material per unit", table_id: file_0_view_0, key: "Product Name"
- $p_i$: "Selling Price", table_id: file_0_view_0, key: "Product Name"
- $v_i$: "Variable Cost", table_id: file_0_view_0, key: "Product Name"
- $L$: $1650$ (from user description)
- $M$: $1850$ (from user description)
- $F$: $4500$ (from user description)
- $x_i$: Production quantity of product $i$, continuous, $\geq 0$ for all $i \in I$