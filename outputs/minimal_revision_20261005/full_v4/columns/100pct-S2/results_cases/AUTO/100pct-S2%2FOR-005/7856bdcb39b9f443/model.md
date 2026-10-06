**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of bread types, indexed by $i$. (from file_1_view_0, column item_name)

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$. (from file_1_view_0, column item_value)
- $a_i$: Storage space required per unit of bread type $i$. (from file_1_view_0, column resource_requirement)
- $C$: Total available storage capacity. (from file_0_view_0, column resource_capacity)

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All records in file_1_view_0, column item_name
- $v_i$: file_1_view_0, column item_value, keyed by item_name
- $a_i$: file_1_view_0, column resource_requirement, keyed by item_name
- $C$: file_0_view_0, column resource_capacity, source_row 0

---

**Summary:**  
Choose integer quantities $x_i$ of each bread type $i$ to maximize total expected profit, subject to the total storage used not exceeding $C$. All parameters and index sets are mapped directly from the supplied CSV data.