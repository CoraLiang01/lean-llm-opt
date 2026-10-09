### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with ProductName from file_1_view_0).

**Decision variables:**
- $x_i$: number of units of product $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

**Parameters:**
- $v_i$: Value of product $i$ (from Value in file_1_view_0)
- $w_i$: Weight of product $i$ (from Weight in file_1_view_0)
- $C$: Overall stock capacity (from Capacity in file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All ProductName in file_1_view_0
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0
- $x_i$: Decision variable for each ProductName in file_1_view_0

All parameters and index sets are mapped directly from the current CSV data as described above.