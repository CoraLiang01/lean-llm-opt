#### Mathematical Model

Let:
- $I$ = set of suppliers, from file_1_view_0: $\{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10\}$
- $J$ = set of customer groups, from file_0_view_0: $\{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\}$
- $d_j$ = demand of customer $j \in J$, from file_0_view_0
- $s_i$ = supply capacity of supplier $i \in I$, from file_1_view_0
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$, from file_2_view_0
- $x_{ij} \geq 0$ = quantity shipped from supplier $i$ to customer $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (suppliers): All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customers): All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand column in file_0_view_0, indexed by customer_id.
- $s_i$: supply_capacity column in file_1_view_0, indexed by supplier_id.
- $c_{ij}$: transportation_cost_to_Ck columns in file_2_view_0 (transportation_costs.csv), with $i$ = supplier_id, $j$ = customer_id, using the column whose suffix matches $j$.
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$.

All sets, parameters, and indices are defined exactly as in the current CSV data, preserving source order and identifiers.