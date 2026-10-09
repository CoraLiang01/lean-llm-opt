#### Mathematical Model

Let:
- $I$ = set of platforms (indexed by $i$), with business identifier resource_id from file_0_view_0.
- $J$ = set of game genres (indexed by $j$), with business identifier item_name from file_1_view_0.

Parameters:
- $c_i$ = resource_capacity of platform $i$ (from file_0_view_0, column resource_capacity).
- $v_j$ = item_value of genre $j$ (from file_1_view_0, column item_value).
- $a_j$ = resource_requirement of genre $j$ (from file_1_view_0, column resource_requirement).

Decision variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

- $I$: All resource_id in file_0_view_0 (capacity.csv), column resource_id.
- $J$: All item_name in file_1_view_0 (products.csv), column item_name.
- $c_i$: file_0_view_0, column resource_capacity, keyed by resource_id.
- $v_j$: file_1_view_0, column item_value, keyed by item_name.
- $a_j$: file_1_view_0, column resource_requirement, keyed by item_name.
- $x_{ij}$: Integer, $\geq 0$, for all $i \in I$, $j \in J$.