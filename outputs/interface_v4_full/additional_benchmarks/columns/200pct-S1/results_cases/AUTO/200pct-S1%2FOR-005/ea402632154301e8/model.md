#### Abstract Mathematical Model

**Sets:**
- $I$: Set of bread types, indexed by $i$ (from products.csv, column item_name).

**Parameters:**
- $v_i$: Expected profit per unit of bread type $i$ (from products.csv, column item_value).
- $a_i$: Storage requirement per unit of bread type $i$ (from products.csv, column resource_requirement).
- $C$: Total available storage capacity (from capacity.csv, column resource_capacity).

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day. ($x_i \in \mathbb{Z}_{\geq 0}$)

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

#### Data Mapping

- $I$ (bread types): file_1_view_0.item_name
- $v_i$: file_1_view_0.item_value (for each $i$)
- $a_i$: file_1_view_0.resource_requirement (for each $i$)
- $C$: file_0_view_0.resource_capacity

Each $x_i$ is the number of units of bread type $i$ to order, as indexed by file_1_view_0.item_name. The storage constraint uses file_1_view_0.resource_requirement and file_0_view_0.resource_capacity. The objective coefficients are from file_1_view_0.item_value. All variables are nonnegative integers.