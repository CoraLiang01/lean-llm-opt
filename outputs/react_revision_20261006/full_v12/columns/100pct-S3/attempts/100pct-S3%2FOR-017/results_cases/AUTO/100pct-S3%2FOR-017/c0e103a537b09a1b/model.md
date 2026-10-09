#### Mathematical Model

Let $S$ be the set of suppliers (from "supplier_id" in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from "customer_id" in customer_demand.csv and transportation_costs.csv).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in S$ to customer $j \in C$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from "demand" in customer_demand.csv).
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity" in supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv).

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in C
\]
\[
\sum_{j \in C} x_{ij} \leq s_i \qquad \forall i \in S
\]
\[
x_{ij} \geq 0 \qquad \forall i \in S,\, j \in C
\]

#### Data Mapping

- $S$ (suppliers): All "supplier_id" in file_1_view_0 (supply_capacity.csv) and file_2_view_0 (transportation_costs.csv), source order.
- $C$ (customers): All "customer_id" in file_0_view_0 (customer_demand.csv) and columns "transportation_cost_to_Ck" in file_2_view_0 (transportation_costs.csv), source order.
- $d_j$: "demand" column in file_0_view_0, indexed by "customer_id".
- $s_i$: "supply_capacity" column in file_1_view_0, indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" columns in file_2_view_0, row "supplier_id" $i$, column for customer $j$.

All sets, indices, and parameters are defined by the current CSV data, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.