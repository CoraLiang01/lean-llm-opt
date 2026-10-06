**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from all resource_id in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all item_name in file_1_view_0)

**Parameters:**
- $v_p$: value of one unit of product $p$ (item_value from file_1_view_0)
- $w_p$: space/weight required by one unit of product $p$ (resource_requirement from file_1_view_0)
- $C_s$: capacity of shelf $s$ (resource_capacity from file_0_view_0)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ allocated to shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

- $S$: All resource_id in file_0_view_0 (capacity.csv)
- $P$: All item_name in file_1_view_0 (products.csv)
- $C_s$: resource_capacity from file_0_view_0, indexed by resource_id
- $v_p$: item_value from file_1_view_0, indexed by item_name
- $w_p$: resource_requirement from file_1_view_0, indexed by item_name

**Variable Mapping**

- $x_{sp}$: Number of units of product $p$ (item_name in file_1_view_0) allocated to shelf $s$ (resource_id in file_0_view_0)

---

**Summary:**  
Maximize total value of products allocated to shelves, subject to each shelf's capacity, with integer, nonnegative allocations. All parameters and index sets are mapped directly to the supplied data columns and business identifiers.