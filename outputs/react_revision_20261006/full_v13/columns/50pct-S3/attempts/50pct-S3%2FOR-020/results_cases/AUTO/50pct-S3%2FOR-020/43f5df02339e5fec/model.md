#### Mathematical Model

Let:
- $I$ = set of warehouses (indexed by $i$), from supplier_id in supply_capacity.csv and transportation_costs.csv: $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$ = set of stores (indexed by $j$), from customer_id in customer_demand.csv and transportation_costs.csv: $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$
- $d_j$ = demand (units) for store $j$ (from customer_demand.csv)
- $s_i$ = supply capacity (units) for warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Store demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Warehouse supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (warehouses): supplier_id from supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0)
- $J$ (stores): customer_id from customer_demand.csv (file_0_view_0) and columns transportation_cost_to_D* in transportation_costs.csv (file_2_view_0)
- $d_j$: demand_units from customer_demand.csv (file_0_view_0), column "demand_units", indexed by "customer_id"
- $s_i$: supply_capacity_units from supply_capacity.csv (file_1_view_0), column "supply_capacity_units", indexed by "supplier_id"
- $c_{ij}$: transportation_cost_to_D* columns from transportation_costs.csv (file_2_view_0), row "supplier_id", column "transportation_cost_to_Dk" where $j$ corresponds to $Dk$
- $x_{ij}$: decision variable, continuous, nonnegative, for all $i \in I$, $j \in J$