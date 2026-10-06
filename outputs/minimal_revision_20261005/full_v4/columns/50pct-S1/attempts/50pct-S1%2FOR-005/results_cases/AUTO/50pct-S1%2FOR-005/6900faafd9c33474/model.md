**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of bread types, indexed by $i$ (from all rows in file_1_view_0, column item_name).

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$ (file_1_view_0, column item_value, indexed by item_name).
- $a_i$: Storage space required per unit of bread type $i$ (file_1_view_0, column resource_requirement, indexed by item_name).
- $C$: Total available storage capacity (file_0_view_0, column resource_capacity).

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

- $I$: All records in file_1_view_0, column item_name.
- $v_i$: file_1_view_0, column item_value, indexed by item_name.
- $a_i$: file_1_view_0, column resource_requirement, indexed by item_name.
- $C$: file_0_view_0, column resource_capacity.

**Notes:**
- All bread types from products.csv are included.
- Storage capacity is from capacity.csv.
- All variables are integer and nonnegative as required.