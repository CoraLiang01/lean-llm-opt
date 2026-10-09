##### Mathematical Model

Let:
- $I$ = set of production plants = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (from supply_capacity.csv)
- $J$ = set of retail outlets = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from customer_demand.csv)
- $x_{ij} \geq 0$ = quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous)
- $d_j$ = demand at outlet $j$ (from customer_demand.csv)
- $s_i$ = supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (plants): supplier_id in supply_capacity.csv (table_id: file_1_view_0)
- $J$ (outlets): customer_id in customer_demand.csv (table_id: file_0_view_0)
- $d_j$: demand column in customer_demand.csv (table_id: file_0_view_0, column: demand, indexed by customer_id)
- $s_i$: supply_capacity column in supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity, indexed by supplier_id)
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv (table_id: file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C4, indexed by supplier_id and customer_id as mapped in relationships)

All index sets, parameters, and constraints are defined directly from the current CSV data and user query. No data is omitted or invented.