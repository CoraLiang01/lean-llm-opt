**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products, indexed by $i$ (from all ProductName in file_1_view_0).

**Parameters:**
- $v_i$: Value per unit of product $i$ (from Value in file_1_view_0).
- $w_i$: Weight per unit of product $i$ (from Weight in file_1_view_0).
- $C$: Total stock capacity (from Capacity in file_0_view_0, row 0).

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

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

**Data Mapping**

- $I$: All records in file_1_view_0, column ProductName.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, row 0, column Capacity.