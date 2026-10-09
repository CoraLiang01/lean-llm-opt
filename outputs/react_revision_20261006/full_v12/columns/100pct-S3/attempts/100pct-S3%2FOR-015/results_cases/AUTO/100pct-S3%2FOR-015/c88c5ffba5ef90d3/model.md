### Mathematical Model

**Sets:**
- $S$: set of shelves (indexed by $s$), with shelf IDs given by file_0_view_0.resource_id
- $P$: set of products (indexed by $p$), with product IDs given by file_1_view_0.item_name

**Parameters:**
- $v_p$: value of one unit of product $p$ (file_1_view_0.item_value)
- $w_p$: weight (resource requirement) of one unit of product $p$ (file_1_view_0.resource_requirement)
- $C_s$: capacity of shelf $s$ (file_0_view_0.resource_capacity)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

**Constraints:**
- Shelf capacity for each shelf $s$:
\[
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
\]
- Integrality and nonnegativity:
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

### Data Mapping

- $S$: All shelf IDs from file_0_view_0.resource_id
- $P$: All product IDs from file_1_view_0.item_name
- $v_p$: file_1_view_0.item_value (indexed by item_name)
- $w_p$: file_1_view_0.resource_requirement (indexed by item_name)
- $C_s$: file_0_view_0.resource_capacity (indexed by resource_id)
- $x_{s,p}$: number of units of product $p$ on shelf $s$ (decision variable, integer, $\geq 0$)

All parameters are mapped directly from the corresponding columns and IDs in the provided CSV files.