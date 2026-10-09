#### Mathematical Model

Let $I$ be the set of suppliers (from "supply_capacity.csv"): $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$.

Let $J$ be the set of customer groups (from "customer_demand.csv"): $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$.

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

- $I$ (suppliers): All unique values in column "supplier_id" of table_id file_1_view_0 ("supply_capacity.csv"), source order.
- $J$ (customers): All unique values in column "customer_id" of table_id file_0_view_0 ("customer_demand.csv"), source order.
- $d_j$: For each $j \in J$, value in column "demand" of table_id file_0_view_0, row where "customer_id" = $j$.
- $s_i$: For each $i \in I$, value in column "supply_capacity" of table_id file_1_view_0, row where "supplier_id" = $i$.
- $c_{ij}$: For each $i \in I$, $j \in J$, value in column "transportation_cost_to_$j$" of table_id file_2_view_0 ("transportation_costs.csv"), row where "supplier_id" = $i$.

- Decision variables $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are defined exactly as above, using the full set of entities and coefficients from the current CSV data.