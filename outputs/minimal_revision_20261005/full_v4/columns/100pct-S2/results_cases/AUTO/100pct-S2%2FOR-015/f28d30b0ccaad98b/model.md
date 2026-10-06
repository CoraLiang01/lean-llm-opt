**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from `file_0_view_0.resource_id`)
- $P$: set of products, indexed by $p$ (from `file_1_view_0.item_name`)

**Parameters:**
- $v_p$: value of one unit of product $p$ (from `file_1_view_0.item_value`)
- $w_p$: space/weight required by one unit of product $p$ (from `file_1_view_0.resource_requirement`)
- $C_s$: capacity (space/weight limit) of shelf $s$ (from `file_0_view_0.resource_capacity`)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

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

- $S$: All values of `resource_id` from `file_0_view_0` (capacity.csv)
- $P$: All values of `item_name` from `file_1_view_0` (products.csv)
- $v_p$: `item_value` from `file_1_view_0` (products.csv), keyed by `item_name`
- $w_p$: `resource_requirement` from `file_1_view_0` (products.csv), keyed by `item_name`
- $C_s$: `resource_capacity` from `file_0_view_0` (capacity.csv), keyed by `resource_id`
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$

---

**Summary:**  
Allocate integer units of each product to each shelf to maximize total value, subject to each shelf's capacity. All data and index sets are mapped directly from the provided CSV files and columns.