**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of bread types, indexed by $i$ (from all rows in products.csv, column item_name).

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$ (from products.csv, column item_value, for each $i \in I$).
- $a_i$: Storage requirement per unit of bread type $i$ (from products.csv, column resource_requirement, for each $i \in I$).
- $C$: Total available storage capacity (from capacity.csv, column resource_capacity).

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative, for all $i \in I$)

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

- $I$: All records in table_id = file_1_view_0, column item_name.
- $v_i$: For each $i \in I$, value from table_id = file_1_view_0, column item_value, where item_name = $i$.
- $a_i$: For each $i \in I$, value from table_id = file_1_view_0, column resource_requirement, where item_name = $i$.
- $C$: Value from table_id = file_0_view_0, column resource_capacity.