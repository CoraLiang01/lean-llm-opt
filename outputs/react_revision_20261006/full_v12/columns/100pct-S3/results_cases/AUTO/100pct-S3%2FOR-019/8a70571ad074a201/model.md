#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from column "supplier_id" in supply_capacity.csv and transportation_costs.csv, and $J$ the set of customer groups from column "customer_id" in customer_demand.csv and the cost matrix columns.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv, column "demand")
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv, columns "transportation_cost_to_demand1", ..., "transportation_cost_to_demand8")

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

- $I$: All "supplier_id" in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $J$: All "customer_id" in customer_demand.csv (table_id: file_0_view_0) and cost matrix columns in transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: customer_demand.csv (table_id: file_0_view_0), column "demand", indexed by "customer_id"
- $s_i$: supply_capacity.csv (table_id: file_1_view_0), column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0), row "supplier_id", columns "transportation_cost_to_demand1", ..., "transportation_cost_to_demand8" (column_id_mapping in Observation), mapped to $i$ and $j$ via relationships in Observation

All indices, parameters, and constraints are defined exactly as in the current source data.