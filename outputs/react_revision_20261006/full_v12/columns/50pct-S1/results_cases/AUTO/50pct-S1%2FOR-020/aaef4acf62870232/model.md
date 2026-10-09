#### Mathematical Model

Let $I$ be the set of warehouses (suppliers) and $J$ the set of stores (customers):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand at store $j$ (units)
- $s_i$: supply capacity at warehouse $i$ (units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

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

#### Data Mapping

- $I$ (warehouses): supplier_id from supply_capacity.csv and transportation_costs.csv
- $J$ (stores): customer_id from customer_demand.csv and destination columns in transportation_costs.csv
- $d_j$: file_0_view_0, column demand_units, key customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity_units, key supplier_id $i$
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (e.g., transportation_cost_to_D1 for $j$=D1)

All index sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and bounds.