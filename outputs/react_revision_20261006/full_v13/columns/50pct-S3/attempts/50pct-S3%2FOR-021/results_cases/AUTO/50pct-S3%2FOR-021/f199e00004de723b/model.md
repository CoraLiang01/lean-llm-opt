Mathematical Model

Sets:
- $I$: set of production plants (indexed by $i$), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$: set of retail outlets (indexed by $j$), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Parameters:
- $d_j$: demand of retail outlet $j$ (from file_0_view_0, column demand)
- $s_i$: production capacity of plant $i$ (from file_1_view_0, column supply_capacity)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, columns transportation_cost_to_C1, ..., _C4)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each outlet:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity for each plant:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

Data Mapping

- $I$ (plants): supplier_id from file_1_view_0 and file_2_view_0, in source order: S1, S2, S3, S4
- $J$ (outlets): customer_id from file_0_view_0 and columns transportation_cost_to_C1, ..., _C4 in file_2_view_0, in source order: C1, C2, C3, C4
- $d_j$: file_0_view_0, column demand, for each customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity, for each supplier_id $i$
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (e.g., transportation_cost_to_C1 for $j$=C1)

All indices, parameters, and constraints are mapped directly to the current source data as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.