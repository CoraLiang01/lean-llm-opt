#### Abstract Mathematical Model

Let $I$ be the set of bread types (indexed by $i$).

**Parameters:**
- $p_i$: expected profit per unit of bread type $i$ (from file_1_view_0, column item_value, indexed by item_name)
- $a_i$: storage space required per unit of bread type $i$ (from file_1_view_0, column resource_requirement, indexed by item_name)
- $C$: total storage capacity (from file_0_view_0, column resource_capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread type $i$ to order each day

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
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

- $p_i$: file_1_view_0, column item_value, indexed by item_name
- $a_i$: file_1_view_0, column resource_requirement, indexed by item_name
- $C$: file_0_view_0, column resource_capacity

All data is used in source order and with original identifiers.