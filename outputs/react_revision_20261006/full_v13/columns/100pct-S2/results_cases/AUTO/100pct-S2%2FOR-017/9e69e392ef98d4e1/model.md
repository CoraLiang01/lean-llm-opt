#### Mathematical Model

Let:
- $I$ = set of suppliers, indexed by $i$ (from column "supplier_id" in supply_capacity.csv and transportation_costs.csv)
- $J$ = set of customer groups, indexed by $j$ (from column "customer_id" in customer_demand.csv and transportation_costs.csv)
- $d_j$ = demand of customer $j$ (from "demand" in customer_demand.csv)
- $s_i$ = supply capacity of supplier $i$ (from "supply_capacity" in supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from supplier $i$ to customer $j$ (continuous)

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

- $I$ (suppliers): All "supplier_id" in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $J$ (customers): All "customer_id" in customer_demand.csv (table_id: file_0_view_0) and columns "transportation_cost_to_Ck" in transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: "demand" column in customer_demand.csv (table_id: file_0_view_0), indexed by "customer_id"
- $s_i$: "supply_capacity" column in supply_capacity.csv (table_id: file_1_view_0), indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in transportation_costs.csv (table_id: file_2_view_0), with row "supplier_id" = $i$, column suffix $k$ matching $j$
- $x_{ij}$: Decision variable for each $(i,j)$ pair, $i \in I$, $j \in J$

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order. No data is omitted or aggregated.