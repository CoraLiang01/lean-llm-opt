#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from column "supplier_id" in supply_capacity.csv and transportation_costs.csv, and $J$ the set of customer groups from column "customer_id" in customer_demand.csv and transportation_costs.csv.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from "demand_units" in customer_demand.csv)
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity_units" in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_cost_to_{j}" in transportation_costs.csv)

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

- $I$: All "supplier_id" in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id) and transportation_costs.csv (table_id: file_2_view_0, column: supplier_id)
- $J$: All "customer_id" in customer_demand.csv (table_id: file_0_view_0, column: customer_id) and as suffixes in transportation_costs.csv columns "transportation_cost_to_{customer_id}" (table_id: file_2_view_0)
- $d_j$: "demand_units" for customer $j$ in customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$: "supply_capacity_units" for supplier $i$ in supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$: "transportation_cost_to_{j}" for supplier $i$ in transportation_costs.csv (table_id: file_2_view_0, columns: supplier_id, transportation_cost_to_{customer_id})

All index sets and parameters are defined by the full set of returned rows and columns in the current data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.