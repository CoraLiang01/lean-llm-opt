#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from the "supplier_id" column of "supply_capacity.csv" (table_id: file_1_view_0), and $J$ be the set of customer groups from the "customer_id" column of "customer_demand.csv" (table_id: file_0_view_0).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ ("demand_units" in file_0_view_0)
- $s_i$: supply capacity of distribution center $i$ ("supply_capacity_units" in file_1_view_0)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ ("transportation_cost_to_{j}" in file_2_view_0, row "supplier_id" = $i$)

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

- $I$: All "supplier_id" in table_id: file_1_view_0 ("supply_capacity.csv")
- $J$: All "customer_id" in table_id: file_0_view_0 ("customer_demand.csv")
- $d_j$: "demand_units" for customer $j$ in table_id: file_0_view_0
- $s_i$: "supply_capacity_units" for supplier $i$ in table_id: file_1_view_0
- $c_{ij}$: "transportation_cost_to_{j}" for supplier $i$ in table_id: file_2_view_0 ("transportation_costs.csv"), with $j$ mapped as per the relationships in the Observation

All indices, parameters, and constraints are defined directly from the current source data, preserving all identifiers and bounds.