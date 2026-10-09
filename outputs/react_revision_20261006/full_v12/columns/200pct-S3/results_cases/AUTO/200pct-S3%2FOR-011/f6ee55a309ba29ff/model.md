### Mathematical Model

Let $I$ be the set of products, indexed by $i$.

**Parameters:**
- $v_i$: Value per unit of product $i$ (from file_1_view_0, column Value)
- $w_i$: Weight per unit of product $i$ (from file_1_view_0, column Weight)
- $C$: Total stock capacity (from file_0_view_0, column Capacity)

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

- $I$: All products in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (product in file_1_view_0, column ProductName)