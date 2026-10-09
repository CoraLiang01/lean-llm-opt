### Mathematical Model

**Index Sets:**
- $I$: set of bread types, from file_1_view_0[item_name]

**Parameters:**
- $v_i$: expected profit per unit of bread $i$, from file_1_view_0[item_value]
- $a_i$: storage space required per unit of bread $i$, from file_1_view_0[resource_requirement]
- $C$: total storage capacity, from file_0_view_0[resource_capacity]

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread $i$ to order each day

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

### Data Mapping

- $I$: file_1_view_0[item_name]
- $v_i$: file_1_view_0[item_value] (for each $i$)
- $a_i$: file_1_view_0[resource_requirement] (for each $i$)
- $C$: file_0_view_0[resource_capacity]
- $x_i$: number of units of bread $i$ to order each day (decision variable, integer, indexed by $i$)