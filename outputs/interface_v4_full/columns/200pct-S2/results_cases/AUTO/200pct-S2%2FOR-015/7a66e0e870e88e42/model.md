#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity)
- $v_p$: value of product $p$ (from file_1_view_0, column item_value)
- $w_p$: weight (space requirement) of product $p$ (from file_1_view_0, column resource_requirement)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s
\]
- Integrality and nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column resource_id
- $c_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $P$ (products): file_1_view_0, column item_name
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name

All other columns are ignored as they are not referenced in the user query.