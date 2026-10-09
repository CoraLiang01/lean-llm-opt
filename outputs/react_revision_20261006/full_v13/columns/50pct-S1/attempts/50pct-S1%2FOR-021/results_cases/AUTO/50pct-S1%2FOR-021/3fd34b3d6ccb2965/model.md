Mathematical Model

Sets:
- $I$: set of production plants (indexed by $i$), from file_1_view_0.supplier_id = {S1, S2, S3, S4}
- $J$: set of retail outlets (indexed by $j$), from file_0_view_0.customer_id = {C1, C2, C3, C4}

Parameters:
- $d_j$: daily demand at outlet $j$, from file_0_view_0.demand
- $s_i$: daily production capacity at plant $i$, from file_1_view_0.supply_capacity
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$, from file_2_view_0:
    - $c_{i,C1}$ = file_2_view_0.transportation_cost_to_C1
    - $c_{i,C2}$ = file_2_view_0.transportation_cost_to_C2
    - $c_{i,C3}$ = file_2_view_0.transportation_cost_to_C3
    - $c_{i,C4}$ = file_2_view_0.transportation_cost_to_C4

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (for each outlet $j$):
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
2. Supply capacity (for each plant $i$):
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (plants): file_1_view_0.supplier_id
- $J$ (outlets): file_0_view_0.customer_id
- $d_j$: file_0_view_0.demand, indexed by file_0_view_0.customer_id
- $s_i$: file_1_view_0.supply_capacity, indexed by file_1_view_0.supplier_id
- $c_{ij}$: file_2_view_0, indexed by file_2_view_0.supplier_id (rows) and columns transportation_cost_to_C1, transportation_cost_to_C2, transportation_cost_to_C3, transportation_cost_to_C4 (for $j$ = C1, C2, C3, C4 respectively)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$