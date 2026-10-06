**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of bread types, indexed by $i$ (from file_1_view_0, column ProductName)

**Parameters**
- $v_i$: Expected profit per unit of bread type $i$ (file_1_view_0, column Value)
- $w_i$: Storage weight per unit of bread type $i$ (file_1_view_0, column Weight)
- $C$: Total storage capacity of the shop (file_0_view_0, column Capacity)

**Decision Variables**
- $x_i$: Number of units of bread type $i$ to order each day; $x_i \in \mathbb{Z}_{\geq 0}$

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

- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

---

**Notes:**  
- All parameters and index sets are mapped directly to the supplied data columns and file/table IDs.
- The model maximizes total expected profit from bread orders, subject to the bakery's storage capacity, with integer decision variables for each bread type.