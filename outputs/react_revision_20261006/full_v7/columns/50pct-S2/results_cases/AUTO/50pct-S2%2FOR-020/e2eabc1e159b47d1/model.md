##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ be the set of warehouses (indexed by $i$), and $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ be the set of stores (indexed by $j$).

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$ (continuous).

Parameters:
- $d_j$: demand at store $j$ (from customer_demand.csv)
- $s_i$: supply capacity at warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (warehouses): all supplier_id in supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4, S5
- $J$ (stores): all customer_id in customer_demand.csv and transportation_costs.csv: D1, D2, D3, D4, D5
- $d_j$: demand_units from customer_demand.csv, table_id file_0_view_0, column customer_id, value per store
- $s_i$: supply_capacity_units from supply_capacity.csv, table_id file_1_view_0, column supplier_id, value per warehouse
- $c_{ij}$: transportation_cost_to_Dk from transportation_costs.csv, table_id file_2_view_0, row supplier_id $i$, column Dk for $j$

Index sets, parameters, and all coefficients are defined by the current CSV data as described above.