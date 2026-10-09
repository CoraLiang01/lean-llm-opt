#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from "supply_capacity.csv" and $J$ the set of customer groups from "customer_demand.csv". Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ from "transportation_costs.csv".

Subject to:
- Demand satisfaction:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
where $d_j$ is the demand for customer $j$ from "customer_demand.csv".

- Supply capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
where $s_i$ is the supply capacity of supplier $i$ from "supply_capacity.csv".

- Nonnegativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$: All supplier_id in "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id)
- $J$: All customer_id in "customer_demand.csv" (table_id: file_0_view_0, column: customer_id)
- $d_j$: Demand for customer $j$ from "customer_demand.csv" (table_id: file_0_view_0, column: demand_units)
- $s_i$: Supply capacity for supplier $i$ from "supply_capacity.csv" (table_id: file_1_view_0, column: supply_capacity_units)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id})

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order.