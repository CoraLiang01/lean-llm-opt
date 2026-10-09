#### Mathematical Model

Let $I$ be the set of warehouses (from supplier_id in supply_capacity.csv and transportation_costs.csv): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$.

Let $J$ be the set of stores (from customer_id in customer_demand.csv and transportation_costs.csv): $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$.

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
- Demand satisfaction for each store:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
- Supply capacity for each warehouse:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
- Nonnegativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (warehouses): supplier_id in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $J$ (stores): customer_id in customer_demand.csv (table_id: file_0_view_0) and columns transportation_cost_to_D* in transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: demand_units in customer_demand.csv (table_id: file_0_view_0, column: demand_units, key: customer_id)
- $s_i$: supply_capacity_units in supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity_units, key: supplier_id)
- $c_{ij}$: transportation_cost_to_D* columns in transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_D*)

All index sets, parameters, and constraints are mapped directly to the current source data as described above.