#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from column "supplier_id" in supply_capacity.csv and row "supplier_id" in transportation_costs.csv.

Let $J$ be the set of customer groups from column "customer_id" in customer_demand.csv and columns "transportation_cost_to_Ck" in transportation_costs.csv.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from "demand_units" in customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity_units" in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$: All "supplier_id" in supply_capacity.csv (file_1_view_0) and transportation_costs.csv (file_2_view_0)
- $J$: All "customer_id" in customer_demand.csv (file_0_view_0) and columns "transportation_cost_to_Ck" in transportation_costs.csv (file_2_view_0)
- $d_j$: "demand_units" from customer_demand.csv (file_0_view_0), indexed by "customer_id"
- $s_i$: "supply_capacity_units" from supply_capacity.csv (file_1_view_0), indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" from transportation_costs.csv (file_2_view_0), row "supplier_id" $i$, column for customer $j$

Variable domains, objective sense, and all constraints are as specified in the user query and mapped to the current data. No data is omitted or aggregated.