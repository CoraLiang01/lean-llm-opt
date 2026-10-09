#### Mathematical Model

Let $I$ be the set of suppliers (from "supply_capacity.csv"), $J$ the set of customer groups (from "customer_demand.csv"). Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

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

Where:
- $c_{ij}$: unit transportation cost from supplier $i$ to customer $j$ (from "transportation_costs.csv")
- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")

#### Data Mapping

- $I$: All supplier_id in "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id)
- $J$: All customer_id in "customer_demand.csv" (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand for customer $j$ from "customer_demand.csv" (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply_capacity for supplier $i$ from "supply_capacity.csv" (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation_cost_to_Ck for supplier $i$ and customer $j$ from "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck, where Ck matches customer_id in $J$)

Index sets, parameters, and all coefficients are defined exactly as in the current Observation, preserving all identifiers and source order.