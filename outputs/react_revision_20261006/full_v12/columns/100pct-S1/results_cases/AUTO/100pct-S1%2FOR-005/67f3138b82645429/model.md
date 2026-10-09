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

**Constraint:**
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All rows with item_name in file_1_view_0 (products.csv)
- $v_i$: item_value column in file_1_view_0 (products.csv)
- $a_i$: resource_requirement column in file_1_view_0 (products.csv)
- $C$: resource_capacity column in file_0_view_0 (capacity.csv)