**Abstract Mathematical Model**

**Index Sets:**
- $R$: set of shelves (indexed by $r$), from `file_0_view_0.resource_id`
- $I$: set of products (indexed by $i$), from `file_1_view_0.item_name$

**Parameters:**
- $v_i$: value of one unit of product $i$, from `file_1_view_0.item_value`
- $a_i$: space (weight) required by one unit of product $i$, from `file_1_view_0.resource_requirement`
- $C_r$: capacity of shelf $r$, from `file_0_view_0.resource_capacity$

**Decision Variables:**
- $x_{ri}$: number of units of product $i$ to place on shelf $r$; $x_{ri} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{r \in R} \sum_{i \in I} v_i \, x_{ri}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $r \in R$,
   \[
   \sum_{i \in I} a_i \, x_{ri} \leq C_r
   \]

2. **Nonnegativity and Integrality:**  
   For all $r \in R$, $i \in I$,
   \[
   x_{ri} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $R$: All values of `resource_id` from `file_0_view_0` (capacity.csv)
- $I$: All values of `item_name` from `file_1_view_0` (products.csv)
- $v_i$: `file_1_view_0.item_value` (products.csv)
- $a_i$: `file_1_view_0.resource_requirement` (products.csv)
- $C_r$: `file_0_view_0.resource_capacity` (capacity.csv)
- $x_{ri}$: Decision variable for allocation of product $i$ to shelf $r$