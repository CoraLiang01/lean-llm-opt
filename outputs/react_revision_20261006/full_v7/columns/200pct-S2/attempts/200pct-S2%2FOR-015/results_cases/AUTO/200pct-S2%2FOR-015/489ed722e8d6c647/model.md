#### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$, with business identifier resource_id from file_0_view_0.
- $J$ = set of products, indexed by $j$, with business identifier item_name from file_1_view_0.

Parameters:
- $c_i$ = resource_capacity of shelf $i$ (from file_0_view_0, column resource_capacity)
- $v_j$ = item_value of product $j$ (from file_1_view_0, column item_value)
- $a_j$ = resource_requirement of product $j$ (from file_1_view_0, column resource_requirement)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (shelves): file_0_view_0, column resource_id
- $c_i$: file_0_view_0, column resource_capacity, keyed by resource_id
- $J$ (products): file_1_view_0, column item_name
- $v_j$: file_1_view_0, column item_value, keyed by item_name
- $a_j$: file_1_view_0, column resource_requirement, keyed by item_name
- $x_{ij}$: allocation variable for shelf $i$ (resource_id) and product $j$ (item_name)