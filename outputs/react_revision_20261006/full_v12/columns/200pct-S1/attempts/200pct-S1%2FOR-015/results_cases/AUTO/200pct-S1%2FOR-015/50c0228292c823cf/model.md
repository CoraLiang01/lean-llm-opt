#### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from file_0_view_0.resource_id)
- $P$ = set of products, indexed by $p$ (from file_1_view_0.item_name)

Parameters:
- $C_s$ = capacity of shelf $s$ (file_0_view_0.resource_capacity)
- $v_p$ = value of product $p$ (file_1_view_0.item_value)
- $w_p$ = weight (space requirement) of product $p$ (file_1_view_0.resource_requirement)

Decision variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
$$

Subject to:
$$
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
$$

$$
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
$$

---

#### Data Mapping

- $S$ (shelves): file_0_view_0.resource_id
- $C_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $P$ (products): file_1_view_0.item_name
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $w_p$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{s,p}$: number of units of product $p$ on shelf $s$ (decision variable, integer, nonnegative)