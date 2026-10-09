### Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $v_p$: value of one unit of product $p$ (file_1_view_0, column item_value)
- $w_p$: weight (space requirement) of one unit of product $p$ (file_1_view_0, column resource_requirement)
- $C_s$: capacity of shelf $s$ (file_0_view_0, column resource_capacity)

**Decision Variables:**
- $x_{s,p}$: number of units of product $p$ placed on shelf $s$; $x_{s,p} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

**Constraints:**
- Shelf capacity for each shelf $s$:
\[
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \quad \forall s \in S
\]
- Nonnegativity and integrality:
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

---

### Data Mapping

- $S$: All values of file_0_view_0.resource_id
- $P$: All values of file_1_view_0.item_name
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $w_p$: file_1_view_0.resource_requirement, keyed by item_name
- $C_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $x_{s,p}$: Decision variable for each $(s,p)$ pair

All parameters and index sets are defined directly from the current CSV data as specified above.