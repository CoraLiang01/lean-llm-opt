### Mathematical Model

**Sets:**
- $R$: set of shelves (indexed by $r$), with IDs from `file_0_view_0.resource_id`
- $I$: set of products (indexed by $i$), with IDs from `file_1_view_0.item_name`

**Parameters:**
- $v_i$: value of one unit of product $i$ (`file_1_view_0.item_value`)
- $a_i$: space/weight required by one unit of product $i$ (`file_1_view_0.resource_requirement`)
- $c_r$: capacity of shelf $r$ (`file_0_view_0.resource_capacity`)

**Decision Variables:**
- $x_{r,i} \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ placed on shelf $r$

**Objective:**
\[
\max \sum_{r \in R} \sum_{i \in I} v_i \, x_{r,i}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $r \in R$,
   \[
   \sum_{i \in I} a_i \, x_{r,i} \leq c_r
   \]
2. **Nonnegativity and Integrality:**  
   \[
   x_{r,i} \in \mathbb{Z}_{\geq 0} \quad \forall r \in R,\, i \in I
   \]

---

### Data Mapping

- $R$: All `resource_id` in `file_0_view_0` (capacity.csv)
- $I$: All `item_name` in `file_1_view_0` (products.csv)
- $v_i$: `file_1_view_0.item_value` (products.csv)
- $a_i$: `file_1_view_0.resource_requirement` (products.csv)
- $c_r$: `file_0_view_0.resource_capacity` (capacity.csv)
- $x_{r,i}$: Number of units of product $i$ on shelf $r$ (decision variable, indexed by $r$ and $i$)

All indices, parameters, and constraints are mapped directly to the original CSV columns and business IDs.