##### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$, with platform IDs from column "resource_id" in file_0_view_0.
- $J$ = set of game genres, indexed by $j$, with genre names from column "item_name" in file_1_view_0.

Parameters:
- $c_i$ = memory capacity of platform $i$ ("resource_capacity" in file_0_view_0).
- $v_j$ = value per unit of genre $j$ ("item_value" in file_1_view_0).
- $a_j$ = memory requirement per unit of genre $j$ ("resource_requirement" in file_1_view_0).

Decision Variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: All platform records from file_0_view_0, indexed by "resource_id".
- $J$: All genre records from file_1_view_0, indexed by "item_name".
- $c_i$: file_0_view_0, column "resource_capacity", key "resource_id".
- $v_j$: file_1_view_0, column "item_value", key "item_name".
- $a_j$: file_1_view_0, column "resource_requirement", key "item_name".
- $x_{ij}$: Integer, $\geq 0$, for each $(i,j) \in I \times J$.