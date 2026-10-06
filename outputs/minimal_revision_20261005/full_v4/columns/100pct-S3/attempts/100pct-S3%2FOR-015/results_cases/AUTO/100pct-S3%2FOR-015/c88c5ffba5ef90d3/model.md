**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

**Parameters:**
- $v_p$: value of one unit of product $p$ (from file_1_view_0, column item_value)
- $w_p$: space/weight required by one unit of product $p$ (from file_1_view_0, column resource_requirement)
- $C_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$ (shelves): file_0_view_0, column resource_id
- $C_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $P$ (products): file_1_view_0, column item_name
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name

---

**Summary:**  
Maximize the total value of products allocated to shelves, subject to each shelf's capacity, using integer allocation variables for each product-shelf pair. All parameters and index sets are mapped directly to the provided CSV columns and business identifiers.