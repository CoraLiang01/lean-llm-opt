Mathematical Model

Sets:
- $I$: set of production plants (indexed by $i$), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$: set of retail outlets (indexed by $j$), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Parameters (from source data):
- $d_j$: demand at outlet $j$ (from customer_demand.csv, column "demand", table_id file_0_view_0)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv, column "supply_capacity", table_id file_1_view_0)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, columns "transportation_cost_to_C1", ..., table_id file_2_view_0)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction at each outlet:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity at each plant:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Nonnegativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ from supply_capacity.csv (file_1_view_0, column "supplier_id") and transportation_costs.csv (file_2_view_0, column "supplier_id")
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ from customer_demand.csv (file_0_view_0, column "customer_id") and transportation_costs.csv (file_2_view_0, columns "transportation_cost_to_C1", etc.)
- $d_j$ from customer_demand.csv (file_0_view_0, column "demand", indexed by "customer_id")
- $s_i$ from supply_capacity.csv (file_1_view_0, column "supply_capacity", indexed by "supplier_id")
- $c_{ij}$ from transportation_costs.csv (file_2_view_0, columns "transportation_cost_to_C1", ..., indexed by "supplier_id" and mapped to $j$ via the column_axis mapping in the Observation)

All indices, parameters, and constraints are defined exactly as in the current source data. No data is omitted or aggregated. Variable domains and all constraints follow the user query.