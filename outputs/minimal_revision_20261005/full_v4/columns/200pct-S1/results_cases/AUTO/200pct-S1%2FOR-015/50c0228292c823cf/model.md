**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from `file_0_view_0.resource_id`)
- $P$: set of products, indexed by $p$ (from `file_1_view_0.item_name`)

**Parameters:**
- $v_p$: value of one unit of product $p$ (from `file_1_view_0.item_value`)
- $w_p$: space (weight) required by one unit of product $p$ (from `file_1_view_0.resource_requirement`)
- $C_s$: capacity (space limit) of shelf $s$ (from `file_0_view_0.resource_capacity`)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s
   \]

2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All values of `file_0_view_0.resource_id`
- $P$: All values of `file_1_view_0.item_name`
- $v_p$: `file_1_view_0.item_value` (keyed by `item_name`)
- $w_p$: `file_1_view_0.resource_requirement` (keyed by `item_name`)
- $C_s$: `file_0_view_0.resource_capacity` (keyed by `resource_id`)
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$

---

**Summary:**  
Maximize the total value of products allocated to shelves, subject to each shelf's capacity, using integer allocation variables. All parameters and index sets are mapped directly from the provided CSV data using the specified columns.