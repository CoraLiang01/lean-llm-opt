#### Mathematical Model

Let:
- $I$ = set of production plants (from supplier_id in supply_capacity.csv): $\{S1, S2, S3, S4\}$
- $J$ = set of retail outlets (from customer_id in customer_demand.csv): $\{C1, C2, C3, C4\}$
- $x_{ij} \geq 0$ = quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous variable)
- $d_j$ = demand of outlet $j$ (from customer_demand.csv)
- $s_i$ = supply capacity of plant $i$ (from supply_capacity.csv)
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

#### Data Mapping

- $I$ (plants): supplier_id from supply_capacity.csv (table_id: file_1_view_0)
- $J$ (outlets): customer_id from customer_demand.csv (table_id: file_0_view_0)
- $d_j$: demand column in customer_demand.csv (table_id: file_0_view_0, column: demand, indexed by customer_id)
- $s_i$: supply_capacity column in supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity, indexed by supplier_id)
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, columns: transportation_cost_to_C1, ..., transportation_cost_to_C4, mapped to customer_id)

All index sets, parameters, and coefficients are defined exactly as in the current CSV data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.