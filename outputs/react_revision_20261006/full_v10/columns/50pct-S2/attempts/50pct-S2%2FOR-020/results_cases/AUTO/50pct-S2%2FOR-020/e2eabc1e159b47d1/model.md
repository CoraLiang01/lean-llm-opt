Mathematical Model

Sets:
- $S$: set of warehouses (indexed by $i$), from supplier_id in supply_capacity.csv and transportation_costs.csv.
- $D$: set of stores (indexed by $j$), from customer_id in customer_demand.csv and transportation_costs.csv.

Parameters:
- $d_j$: demand (units) for store $j\in D$ (from demand_units in customer_demand.csv).
- $s_i$: supply capacity (units) for warehouse $i\in S$ (from supply_capacity_units in supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_cost_to_Dk in transportation_costs.csv).

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous).

Objective:
\[
\min \sum_{i\in S} \sum_{j\in D} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
\[
\sum_{i\in S} x_{ij} \geq d_j \quad \forall j\in D
\]
2. Supply capacity for each warehouse:
\[
\sum_{j\in D} x_{ij} \leq s_i \quad \forall i\in S
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i\in S,\, j\in D
\]

Data Mapping

Sets:
- $S$ = all supplier_id in supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0): S1, S2, S3, S4, S5
- $D$ = all customer_id in customer_demand.csv (file_0_view_0) and columns D1, D2, D3, D4, D5 in transportation_costs.csv

Parameters:
- $d_j$: demand_units for customer_id $j$ in file_0_view_0 (customer_demand.csv)
- $s_i$: supply_capacity_units for supplier_id $i$ in file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: transportation_cost_to_Dk for supplier_id $i$ and customer_id $j$ in file_2_view_0 (transportation_costs.csv), with column_id_mapping:
  - D1: transportation_cost_to_D1
  - D2: transportation_cost_to_D2
  - D3: transportation_cost_to_D3
  - D4: transportation_cost_to_D4
  - D5: transportation_cost_to_D5

Variables:
- $x_{ij}$: for all $i$ in supplier_id (file_1_view_0), $j$ in customer_id (file_0_view_0)

All indices, parameters, and constraints are mapped exactly to the current source data as described above. No data is omitted or aggregated.