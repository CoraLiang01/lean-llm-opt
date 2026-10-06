**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of bread types, indexed by $i$. (from file_1_view_0, column ProductName)

**Parameters**
- $v_i$: Expected profit per unit of bread type $i$. (file_1_view_0, column Value)
- $w_i$: Storage weight per unit of bread type $i$. (file_1_view_0, column Weight)
- $C$: Total storage capacity of the shop. (file_0_view_0, column Capacity)

**Decision Variables**
- $x_i$: Number of units of bread type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

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

**Summary:**  
Choose integer quantities $x_i$ of each bread type $i$ to maximize total expected profit, subject to the total storage weight not exceeding $C$. All parameters and index sets are mapped directly from the supplied data.