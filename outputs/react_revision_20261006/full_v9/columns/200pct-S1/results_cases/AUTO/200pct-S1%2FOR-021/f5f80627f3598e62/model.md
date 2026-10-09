##### Mathematical Model

Let:
- $I$ = set of production plants (indexed by $i$), from supplier_id in file_1_view_0 and file_2_view_0
- $J$ = set of retail outlets (indexed by $j$), from customer_id in file_0_view_0 and file_2_view_0
- $d_j$ = demand of outlet $j$, from demand in file_0_view_0
- $s_i$ = supply capacity of plant $i$, from supply_capacity in file_1_view_0
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$, from transportation_cost_to_C* in file_2_view_0
- $x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous)

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

##### Data Mapping

- $I$ (plants): supplier_id in file_1_view_0 and file_2_view_0
- $J$ (outlets): customer_id in file_0_view_0 and columns transportation_cost_to_C* in file_2_view_0
- $d_j$: demand column in file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity column in file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_C* columns in file_2_view_0, row supplier_id $i$, column for $j$ as mapped in relationships.column_id_mapping
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$