**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $v_p$: value of one unit of product $p$ (from file_1_view_0, column item_value)
- $w_p$: space/weight required by one unit of product $p$ (from file_1_view_0, column resource_requirement)
- $C_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ allocated to shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$ (shelves): file_0_view_0, column resource_id
- $C_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $P$ (products): file_1_view_0, column item_name
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name

**Variables:**
- $x_{sp}$: number of units of product $p$ on shelf $s$ (indexed by resource_id and item_name)

**All index sets, parameters, and constraints are defined using the exact columns and business identifiers from the supplied data.**