#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity)
- $v_p$: value of product $p$ (from file_1_view_0, column item_value)
- $w_p$: weight (space requirement) of product $p$ (from file_1_view_0, column resource_requirement)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to place on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column resource_id
- $c_s$: file_0_view_0, columns resource_id (shelf key), resource_capacity (capacity)
- $P$ (products): file_1_view_0, column item_name
- $v_p$: file_1_view_0, columns item_name (product key), item_value (value)
- $w_p$: file_1_view_0, columns item_name (product key), resource_requirement (weight/space requirement)
- $x_{sp}$: decision variable for each $(s,p)$ pair

All index sets, parameters, and constraints are mapped directly to the original file columns and business identifiers as required.