**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity, keyed by resource_id)
- $v_p$: value of product $p$ (from file_1_view_0, column item_value, keyed by item_name)
- $a_p$: space/weight requirement of product $p$ (from file_1_view_0, column resource_requirement, keyed by item_name)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} a_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$ (shelves): file_0_view_0, column resource_id
- $c_s$: file_0_view_0, columns resource_id (key), resource_capacity (value)
- $P$ (products): file_1_view_0, column item_name
- $v_p$: file_1_view_0, columns item_name (key), item_value (value)
- $a_p$: file_1_view_0, columns item_name (key), resource_requirement (value)
- $x_{sp}$: decision variable for each $(s,p)$ pair

---

**Notes:**
- All shelves and products from the returned data are included.
- Each shelf's capacity is enforced individually.
- All variables are nonnegative integers as required.