### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with ProductName from file_1_view_0).

**Parameters:**
- $v_i$: Value of product $i$ (from Value in file_1_view_0)
- $w_i$: Weight of product $i$ (from Weight in file_1_view_0)
- $C$: Overall stock capacity (from Capacity in file_0_view_0)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0 (products.csv)
- $w_i$: Weight column in file_1_view_0 (products.csv)
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Number of units of product $i$ to order each day (decision variable, nonnegative integer)