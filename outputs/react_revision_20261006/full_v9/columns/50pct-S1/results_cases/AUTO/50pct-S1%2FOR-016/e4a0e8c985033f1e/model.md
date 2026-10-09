#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from the "supplier_id" column of "supply_capacity.csv" and "transportation_costs.csv".
- $J$ = set of customer groups, indexed by $j$, from the "customer_id" column of "customer_demand.csv" and columns of "transportation_costs.csv".
- $d_j$ = demand (units) for customer group $j$, from "customer_demand.csv".
- $s_i$ = supply capacity (units) for distribution center $i$, from "supply_capacity.csv".
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_costs.csv".
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (distribution centers): All "supplier_id" values in "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id) and "transportation_costs.csv" (table_id: file_2_view_0, column: supplier_id).
- $J$ (customer groups): All "customer_id" values in "customer_demand.csv" (table_id: file_0_view_0, column: customer_id) and all columns with prefix "transportation_cost_to_" in "transportation_costs.csv" (table_id: file_2_view_0).
- $d_j$: "demand_units" from "customer_demand.csv" (table_id: file_0_view_0, columns: customer_id, demand_units).
- $s_i$: "supply_capacity_units" from "supply_capacity.csv" (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units).
- $c_{ij}$: "transportation_cost_to_Ck" columns for each $j$ in "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id $i$, column: transportation_cost_to_Ck for customer $j$).
- $x_{ij}$: Decision variable for each $(i,j)$ pair, continuous and nonnegative.

All index sets, parameters, and constraints are defined exactly as in the current CSV data, preserving all identifiers and bounds.