#### Abstract Mathematical Model

**Sets:**
- $I$: Set of produce types, indexed by $i$ (from file_1_view_0, column ProductName).

**Parameters:**
- $v_i$: Value (benefit) per unit of produce $i$ (file_1_view_0, column Value, keyed by ProductName).
- $w_i$: Weight per unit of produce $i$ (file_1_view_0, column Weight, keyed by ProductName).
- $C$: Overall inventory capacity (file_0_view_0, column Capacity).

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of produce $i$ to order daily.

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

#### Data Mapping

- $I$ (produce types): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

All data is used in original file and row order, with no omitted or synthesized values.