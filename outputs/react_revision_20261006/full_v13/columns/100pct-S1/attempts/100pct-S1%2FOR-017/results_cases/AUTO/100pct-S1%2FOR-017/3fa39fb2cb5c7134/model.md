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

- $I$ (suppliers): All "supplier_id" in table_id="file_1_view_0" (supply_capacity.csv), source order.
- $J$ (customers): All "customer_id" in table_id="file_0_view_0" (customer_demand.csv), source order.
- $d_j$: "demand" for customer $j$ in table_id="file_0_view_0", column "demand", indexed by "customer_id".
- $s_i$: "supply_capacity" for supplier $i$ in table_id="file_1_view_0", column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" for supplier $i$ and customer $j$ in table_id="file_2_view_0" (transportation_costs.csv), columns "transportation_cost_to_C1"..."transportation_cost_to_C10", indexed by "supplier_id" and mapped to $j$ via the column_axis mapping in the Observation.

All index sets, parameters, and coefficients are defined exactly as in the current Observation, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.