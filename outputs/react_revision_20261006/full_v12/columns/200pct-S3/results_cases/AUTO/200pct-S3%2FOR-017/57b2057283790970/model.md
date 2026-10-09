#### Mathematical Model

Let $I$ be the set of suppliers (from "supply_capacity.csv" and "transportation_costs.csv"): $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$.

Let $J$ be the set of customer groups (from "customer_demand.csv" and "transportation_costs.csv"): $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

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

- $I$ (suppliers): all "supplier_id" in table_id="file_1_view_0" and "file_2_view_0"
- $J$ (customers): all "customer_id" in table_id="file_0_view_0" and columns with suffix in "transportation_cost_to_*" in table_id="file_2_view_0"
- $d_j$: "demand" column in table_id="file_0_view_0", indexed by "customer_id"
- $s_i$: "supply_capacity" column in table_id="file_1_view_0", indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in table_id="file_2_view_0", row "supplier_id" $i$, column for customer $j$ (see relationships.matrix_table_id="file_2_view_0", row_axis="supplier_id", column_axis="customer_id")

All indices, parameters, and constraints are defined directly from the current source data, preserving all identifiers and bounds.