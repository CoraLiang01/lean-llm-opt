**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of drug products, indexed by $i$ (from all ProductName in file_1_view_0).

**Parameters:**
- $v_i$: Value (benefit) per unit of drug $i$ (from Value in file_1_view_0, indexed by ProductName).
- $w_i$: Weight (stock space required) per unit of drug $i$ (from Weight in file_1_view_0, indexed by ProductName).
- $C$: Total stock capacity (from Capacity in file_0_view_0).

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of drug $i$ to order each day.

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
- $v_i$: file_1_view_0, columns ProductName and Value.
- $w_i$: file_1_view_0, columns ProductName and Weight.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.

---

**Summary:**  
Choose integer order quantities $x_i$ for each drug $i$ to maximize total benefit, subject to the overall stock capacity. All parameters and index sets are mapped directly from the supplied CSV data.