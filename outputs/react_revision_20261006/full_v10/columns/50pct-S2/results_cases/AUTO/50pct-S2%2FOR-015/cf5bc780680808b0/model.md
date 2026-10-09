Mathematical Model

Index Sets:
- Let S be the set of shelves, indexed by s, with shelf identifiers resource_id from file_0_view_0.
- Let P be the set of products, indexed by p, with product identifiers item_name from file_1_view_0.

Parameters:
- v_p: value of product p (item_value from file_1_view_0)
- w_p: weight (space requirement) of product p (resource_requirement from file_1_view_0)
- C_s: capacity of shelf s (resource_capacity from file_0_view_0)

Decision Variables:
- x_{s,p}: number of units of product p allocated to shelf s; x_{s,p} ∈ ℤ₊ (nonnegative integers), ∀ s ∈ S, p ∈ P

Objective:
Maximize total value allocated across all shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
$$

Subject to:

Shelf capacity constraints (for each shelf s):
$$
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \quad \forall s \in S
$$

Nonnegativity and integrality:
$$
x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

Data Mapping

Index Sets:
- S (shelves): resource_id from file_0_view_0 (capacity.csv)
- P (products): item_name from file_1_view_0 (products.csv)

Parameters:
- v_p: file_1_view_0, column item_value, key item_name
- w_p: file_1_view_0, column resource_requirement, key item_name
- C_s: file_0_view_0, column resource_capacity, key resource_id

Decision Variables:
- x_{s,p}: number of units of product p (item_name) on shelf s (resource_id), integer, ≥ 0

All constraints and the objective use these mappings and index sets exactly as defined above.