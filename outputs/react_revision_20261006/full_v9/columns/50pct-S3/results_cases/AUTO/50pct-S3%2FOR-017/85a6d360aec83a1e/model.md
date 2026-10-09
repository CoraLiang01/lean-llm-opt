##### Mathematical Model

Let $S$ be the set of suppliers (from "supplier_id" in supply_capacity.csv and transportation_costs.csv), and $C$ the set of customer groups (from "customer_id" in customer_demand.csv and transportation_costs.csv).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in S$ to customer $j \in C$.

Parameters:
- $d_j$: demand of customer $j$ (from "demand" in customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity" in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv)

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in S} x_{ij} \geq d_j \quad \forall j \in C
\]
\[
\sum_{j \in C} x_{ij} \leq s_i \quad \forall i \in S
\]
\[
x_{ij} \geq 0 \quad \forall i \in S,\, j \in C
\]

##### Data Mapping

- $S$ (suppliers): all "supplier_id" in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $C$ (customers): all "customer_id" in customer_demand.csv (table_id: file_0_view_0) and all columns with suffix "transportation_cost_to_Ck" in transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: "demand" column in customer_demand.csv (table_id: file_0_view_0), indexed by "customer_id"
- $s_i$: "supply_capacity" column in supply_capacity.csv (table_id: file_1_view_0), indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in transportation_costs.csv (table_id: file_2_view_0), with row "supplier_id" and column mapped to customer $j$ via the relationships mapping in the Observation

All index sets, parameters, and coefficients are defined exactly as in the current CSV data and relationships. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.