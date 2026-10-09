#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand (units) for store $j \in J$
- $s_i$: supply capacity (units) for warehouse $i \in I$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

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
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (warehouses): All unique values in column "supplier_id" of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (stores): All unique values in column "customer_id" of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: "demand_units" from table_id file_0_view_0, indexed by "customer_id"
- $s_i$: "supply_capacity_units" from table_id file_1_view_0, indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Dk" columns in table_id file_2_view_0 (transportation_costs.csv), with row index "supplier_id" and column index mapped to "customer_id" as per relationships in the Observation

- $x_{ij}$: Decision variable for each $(i,j) \in I \times J$ (continuous, nonnegative)