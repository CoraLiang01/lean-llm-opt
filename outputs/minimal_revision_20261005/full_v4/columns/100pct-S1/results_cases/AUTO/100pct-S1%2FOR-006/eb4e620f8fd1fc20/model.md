**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$. (All ProductName in file_1_view_0)

**Parameters:**
- $v_i$: Value (benefit) of ordering one unit of vehicle type $i$. (file_1_view_0, column Value)
- $w_i$: Weight (inventory space required) for one unit of vehicle type $i$. (file_1_view_0, column Weight)
- $C$: Total inventory capacity. (file_0_view_0, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

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
- $C$: file_0_view_0, column Capacity

---

**Summary:**  
Choose integer order quantities $x_i$ for each vehicle type $i$ to maximize total benefit, subject to the total inventory capacity. All parameters and index sets are mapped directly to the provided data columns and files.