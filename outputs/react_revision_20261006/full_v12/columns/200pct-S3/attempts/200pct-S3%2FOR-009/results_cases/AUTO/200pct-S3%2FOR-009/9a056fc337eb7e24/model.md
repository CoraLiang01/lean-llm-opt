## Mathematical Model

Let $I$ be the set of areas (from ProductName in products.csv).

**Parameters:**
- $v_i$: Value (development benefit) of area $i$ (from Value in products.csv, table_id: file_1_view_0)
- $w_i$: Weight (resource requirement) of area $i$ (from Weight in products.csv, table_id: file_1_view_0)
- $C$: Overall development capacity (from Capacity in capacity.csv, table_id: file_0_view_0)

**Decision Variables:**
- $x_i \geq 0$, integer: scale of development per day in area $i$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

## Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: scale of development per day in area $i$ (decision variable, nonnegative integer, indexed by ProductName in file_1_view_0)