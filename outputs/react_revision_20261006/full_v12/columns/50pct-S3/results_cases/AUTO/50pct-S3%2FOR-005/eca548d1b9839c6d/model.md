### Mathematical Model

Let $I$ be the set of bread types, indexed by $i$ (with item_name from file_1_view_0).

**Parameters:**
- $v_i$: expected profit per unit of bread $i$ (item_value, file_1_view_0)
- $a_i$: storage space required per unit of bread $i$ (resource_requirement, file_1_view_0)
- $C$: total storage capacity (resource_capacity, file_0_view_0)

**Decision Variables:**
- $x_i$: number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All item_name in file_1_view_0
- $v_i$: item_value from file_1_view_0, keyed by item_name
- $a_i$: resource_requirement from file_1_view_0, keyed by item_name
- $C$: resource_capacity from file_0_view_0, row 0
- $x_i$: integer variable for each $i \in I$ (item_name, file_1_view_0)