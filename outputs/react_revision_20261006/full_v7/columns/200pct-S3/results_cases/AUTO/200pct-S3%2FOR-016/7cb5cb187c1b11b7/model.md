##### Decision Variables

For each distribution center $i \in I$ and customer group $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

##### Parameters

- $I$: set of distribution centers (from "supplier_id" in supply_capacity.csv and transportation_costs.csv)
- $J$: set of customer groups (from "customer_id" in customer_demand.csv and transportation_costs.csv)
- $d_j$: demand of customer group $j$ (from "demand_units" in customer_demand.csv)
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity_units" in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv, with $k$ matching $j$)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity** (each distribution center does not exceed its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ = all "supplier_id" in supply_capacity.csv and transportation_costs.csv (S1, S2, ..., S18)
- $J$ = all "customer_id" in customer_demand.csv and transportation_costs.csv (C1, C2, ..., C18)
- $d_j$ = "demand_units" for customer $j$ in customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$ = "supply_capacity_units" for supplier $i$ in supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$ = "transportation_cost_to_Ck" for supplier $i$ and customer $j$ in transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck where $k$ matches $j$)

All index sets, parameters, and coefficients are defined exactly as in the retrieved data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.