**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products (from file_1_view_0, column ProductName)

**Parameters**
- $v_i$: Value per unit of product $i$ (file_1_view_0, column Value)
- $w_i$: Weight per unit of product $i$ (file_1_view_0, column Weight)
- $C$: Total stock capacity (file_0_view_0, column Capacity)

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day

**Objective**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

---

**Notes**
- All products and their parameters are taken directly from the returned rows of file_1_view_0.
- The total stock capacity $C$ is taken from file_0_view_0.
- The model maximizes total value subject to the overall stock capacity.
- All variables are nonnegative integers as required.