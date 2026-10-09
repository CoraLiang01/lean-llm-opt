### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from all resource_id in file_0_view_0)
- $P$ = set of products, indexed by $p$ (from all item_name in file_1_view_0)
- $x_{sp}$ = number of units of product $p$ placed on shelf $s$ (decision variable, integer, $\geq 0$)
- $v_p$ = value of product $p$ (item_value from file_1_view_0)
- $a_p$ = resource requirement (weight) of product $p$ (resource_requirement from file_1_view_0)
- $C_s$ = capacity of shelf $s$ (resource_capacity from file_0_view_0)

#### Objective:
Maximize total value across all shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

#### Constraints:
Shelf capacity for each shelf $s$:
$$
\sum_{p \in P} a_p \, x_{sp} \leq C_s \quad \forall s \in S
$$

Nonnegativity and integrality:
$$
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

---

### Data Mapping

- $S$: All resource_id in file_0_view_0 (capacity.csv, column: resource_id)
- $P$: All item_name in file_1_view_0 (products.csv, column: item_name)
- $v_p$: file_1_view_0, column: item_value, keyed by item_name
- $a_p$: file_1_view_0, column: resource_requirement, keyed by item_name
- $C_s$: file_0_view_0, column: resource_capacity, keyed by resource_id
- $x_{sp}$: Decision variable for each $(s,p)$ pair

All indices, parameters, and constraints are mapped directly to the columns and keys as specified above.