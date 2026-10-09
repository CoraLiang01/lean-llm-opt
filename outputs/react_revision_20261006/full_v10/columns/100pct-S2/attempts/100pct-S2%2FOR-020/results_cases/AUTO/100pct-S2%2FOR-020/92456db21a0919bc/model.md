Mathematical Model

Sets:
Let $I$ be the set of warehouses (indexed by $i$), with $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$.
Let $J$ be the set of stores (indexed by $j$), with $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$.

Parameters:
$d_j$: daily demand at store $j \in J$ (from file_0_view_0, column demand_units, key customer_id).
$s_i$: daily supply capacity at warehouse $i \in I$ (from file_1_view_0, column supply_capacity_units, key supplier_id).
$c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns transportation_cost_to_D1, ..., transportation_cost_to_D5, key supplier_id).

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each store:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]

2. Supply capacity at each warehouse:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]

3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (warehouses): supplier_id from file_1_view_0 and file_2_view_0, in source order.
- $J$ (stores): customer_id from file_0_view_0 and columns transportation_cost_to_D1, ..., transportation_cost_to_D5 in file_2_view_0, in source order.
- $d_j$: demand_units from file_0_view_0, key customer_id.
- $s_i$: supply_capacity_units from file_1_view_0, key supplier_id.
- $c_{ij}$: transportation_cost_to_Dk from file_2_view_0, where $i$ = supplier_id, $j$ = Dk.
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$.

All indices, parameters, and constraints are mapped directly to the current source data as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.