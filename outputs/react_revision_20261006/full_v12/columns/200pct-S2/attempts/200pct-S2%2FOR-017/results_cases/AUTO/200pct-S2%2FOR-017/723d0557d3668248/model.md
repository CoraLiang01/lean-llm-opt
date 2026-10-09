#### Mathematical Model

Let $S$ be the set of suppliers (from column "supplier_id" in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from column "customer_id" in customer_demand.csv and transportation_costs.csv).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in S$ to customer $j \in C$ (continuous).

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv).

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in C
   \]
   where $d_j$ is the demand of customer $j$ (from customer_demand.csv).

2. Supply capacity:
   \[
   \sum_{j \in C} x_{ij} \leq s_i \qquad \forall i \in S
   \]
   where $s_i$ is the supply capacity of supplier $i$ (from supply_capacity.csv).

3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in S,\, j \in C
   \]

#### Data Mapping

- $S$: All "supplier_id" in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0).
- $C$: All "customer_id" in customer_demand.csv (table_id: file_0_view_0) and columns with suffix in transportation_costs.csv (table_id: file_2_view_0).
- $d_j$: "demand" column in customer_demand.csv (table_id: file_0_view_0), indexed by "customer_id".
- $s_i$: "supply_capacity" column in supply_capacity.csv (table_id: file_1_view_0), indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" columns in transportation_costs.csv (table_id: file_2_view_0), with $i$ from "supplier_id" and $j$ from column suffix.

All sets, parameters, and indices are defined exactly as in the current CSV data. No data is omitted or aggregated.