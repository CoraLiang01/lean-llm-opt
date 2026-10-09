#### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with ProductName from products.csv).

**Parameters:**
- $v_i$: Value per unit of product $i$ (from Value in products.csv)
- $w_i$: Weight per unit of product $i$ (from Weight in products.csv)
- $C$: Overall stock capacity (from Capacity in capacity.csv)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$