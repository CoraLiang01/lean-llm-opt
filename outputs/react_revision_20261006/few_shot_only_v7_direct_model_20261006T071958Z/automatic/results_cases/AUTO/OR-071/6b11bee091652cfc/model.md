ABSTRACT MATHEMATICAL MODEL

**Index Sets:**
- $P$: set of products, indexed by $p$ (from all "Product Name" in file_0_view_0)

**Parameters:**
- $l_p$: labor required per unit of product $p$ ("Labor per unit" from file_0_view_0)
- $m_p$: material required per unit of product $p$ ("Material per unit" from file_0_view_0)
- $s_p$: selling price per unit of product $p$ ("Selling Price" from file_0_view_0)
- $v_p$: variable cost per unit of product $p$ ("Variable Cost" from file_0_view_0)
- $L$: total available labor per week (given as 1,650)
- $M$: total available material per week (given as 1,850)
- $F$: fixed weekly operating cost (given as 4,500)

**Decision Variables:**
- $x_p \geq 0$: quantity of product $p$ to produce (continuous, for all $p \in P$)

**Objective:**
\[
\max \left( \sum_{p \in P} (s_p - v_p) x_p - F \right)
\]

**Constraints:**
\[
\sum_{p \in P} l_p x_p \leq L
\]
\[
\sum_{p \in P} m_p x_p \leq M
\]
\[
x_p \geq 0 \quad \forall p \in P
\]

---

**Data Mapping:**

- $P$: All "Product Name" in table_id=file_0_view_0
- $l_p$: "Labor per unit" in table_id=file_0_view_0, keyed by "Product Name"
- $m_p$: "Material per unit" in table_id=file_0_view_0, keyed by "Product Name"
- $s_p$: "Selling Price" in table_id=file_0_view_0, keyed by "Product Name"
- $v_p$: "Variable Cost" in table_id=file_0_view_0, keyed by "Product Name"
- $L$: 1,650 (from user description)
- $M$: 1,850 (from user description)
- $F$: 4,500 (from user description)
- $x_p$: continuous, nonnegative, for all $p \in P$