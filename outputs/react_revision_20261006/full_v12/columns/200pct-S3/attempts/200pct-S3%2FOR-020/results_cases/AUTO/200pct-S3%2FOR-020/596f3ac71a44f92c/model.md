#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand (units) for store $j$ (from file_0_view_0, column demand_units)
- $s_i$: supply capacity (units) for warehouse $i$ (from file_1_view_0, column supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns transportation_cost_to_D1, ..., transportation_cost_to_D5)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Store demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Warehouse supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Nonnegativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (warehouses): all supplier_id in file_1_view_0 (supply_capacity.csv)
- $J$ (stores): all customer_id in file_0_view_0 (customer_demand.csv)
- $d_j$: demand_units from file_0_view_0, column demand_units, for each customer_id $j$
- $s_i$: supply_capacity_units from file_1_view_0, column supply_capacity_units, for each supplier_id $i$
- $c_{ij}$: transportation_cost_to_Dk from file_2_view_0, where $i$ = supplier_id, $j$ = Dk (column suffix matches customer_id)

All indices, parameters, and coefficients are mapped exactly as in the current Observation. No data is omitted or aggregated.