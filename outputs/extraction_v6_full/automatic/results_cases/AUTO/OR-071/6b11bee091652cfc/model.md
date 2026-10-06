#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$
- $x_i$ = production quantity of product $i$ (continuous, $x_i \geq 0$)

Parameters (for each $i \in I$):
- $l_i$ = labor required per unit of product $i$
- $m_i$ = material required per unit of product $i$
- $s_i$ = selling price per unit of product $i$
- $v_i$ = variable cost per unit of product $i$

Global parameters:
- $L = 1650$ (weekly labor capacity)
- $M = 1850$ (weekly material capacity)
- $F = 4500$ (fixed weekly operating cost)

**Objective:**
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

**Subject to:**
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

#### Data Mapping

- $I$: All rows in 41.csv, column "Product Name", table_id: file_0_view_0
- $l_i$: 41.csv, column "Labor per unit", table_id: file_0_view_0, key: "Product Name"
- $m_i$: 41.csv, column "Material per unit", table_id: file_0_view_0, key: "Product Name"
- $s_i$: 41.csv, column "Selling Price", table_id: file_0_view_0, key: "Product Name"
- $v_i$: 41.csv, column "Variable Cost", table_id: file_0_view_0, key: "Product Name"
- $L$: scalar $1650$ (from user description)
- $M$: scalar $1850$ (from user description)
- $F$: scalar $4500$ (from user description)
- $x_i$: continuous, nonnegative, indexed by $i \in I$ ("Product Name" in file_0_view_0)

---

**Notes:**  
- All 198 products in 41.csv are included, preserving source order and identifiers.
- All parameters are mapped directly to their columns and table_id as required.
- The model maximizes net profit as defined in the user query.