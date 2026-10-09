Mathematical Model

Sets:
- $I$: set of production plants (from supplier_id in supply_capacity.csv and transportation_costs.csv), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$: set of retail outlets (from customer_id in customer_demand.csv and transportation_costs.csv), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Parameters:
- $d_j$: demand of outlet $j \in J$ (from demand in customer_demand.csv)
- $s_i$: supply capacity of plant $i \in I$ (from supply_capacity in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, column mapping: transportation_cost_to_Ck $\rightarrow$ $Ck$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity (each plant ships no more than its capacity):
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (plants): supplier_id from supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4 (file_1_view_0, file_2_view_0)
- $J$ (outlets): customer_id from customer_demand.csv and transportation_costs.csv: C1, C2, C3, C4 (file_0_view_0, file_2_view_0)
- $d_j$: demand from demand column in customer_demand.csv, table_id file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity from supply_capacity column in supply_capacity.csv, table_id file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv, table_id file_2_view_0, row supplier_id $i$, column Ck $j$ (mapping: transportation_cost_to_C1 $\rightarrow$ C1, etc.)

All indices, parameters, and constraints are mapped directly to the current source data as described above. No data is omitted or aggregated. Variable domains and all constraints follow the user query exactly.