**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of produce types (from all ProductName in file_1_view_0)

**Parameters:**
- $v_i$: Value (benefit) per unit of produce $i$ (from Value, file_1_view_0, column ProductName)
- $w_i$: Weight per unit of produce $i$ (from Weight, file_1_view_0, column ProductName)
- $C$: Total inventory capacity (from Capacity, file_0_view_0, row 0)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of produce $i$ to order daily

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

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, row 0, column Capacity

**Variable mapping:** $x_i$ is the daily order quantity for produce $i$ (ProductName from file_1_view_0)