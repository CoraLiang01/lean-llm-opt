Mathematical Model

Sets:
- $I$: set of warehouses (suppliers), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$: set of stores (customers), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand (units) for store $j \in J$ (from file_0_view_0, column demand_units)
- $s_i$: supply capacity (units) for warehouse $i \in I$ (from file_1_view_0, column supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns transportation_cost_to_D1, ..., transportation_cost_to_D5)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity for each warehouse:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (warehouses): supplier_id from file_1_view_0 and file_2_view_0, in source order: S1, S2, S3, S4, S5
- $J$ (stores): customer_id from file_0_view_0 and columns transportation_cost_to_D1, ..., transportation_cost_to_D5 in file_2_view_0, in source order: D1, D2, D3, D4, D5
- $d_j$: demand_units from file_0_view_0, mapped by customer_id
- $s_i$: supply_capacity_units from file_1_view_0, mapped by supplier_id
- $c_{ij}$: transportation_cost_to_Dk from file_2_view_0, mapped by supplier_id (row) and Dk (column, where k=1..5)

All indices, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and source order.