Mathematical Model

Sets:
- $I$: set of production plants (from supplier_id in file_1_view_0): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$: set of retail outlets (from customer_id in file_0_view_0): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Parameters:
- $d_j$: demand of outlet $j \in J$ (from demand in file_0_view_0)
- $s_i$: supply capacity of plant $i \in I$ (from supply_capacity in file_1_view_0)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_cost_to_C* in file_2_view_0, mapped by supplier_id and customer_id)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each outlet:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity for each plant:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (plants): supplier_id from file_1_view_0
- $J$ (outlets): customer_id from file_0_view_0
- $d_j$: demand from file_0_view_0, column demand, indexed by customer_id
- $s_i$: supply_capacity from file_1_view_0, column supply_capacity, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_C* from file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4, indexed by supplier_id (rows) and customer_id (columns)
- $x_{ij}$: decision variable for shipment from $i$ to $j$ (continuous, nonnegative)

All sets, parameters, and mappings are defined exactly as in the current CSV data, preserving all identifiers and source order. No data is omitted or aggregated.