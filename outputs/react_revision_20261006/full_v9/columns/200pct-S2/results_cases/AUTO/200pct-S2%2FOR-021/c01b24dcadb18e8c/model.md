##### Mathematical Model

Let:
- $I$ = set of production plants = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$ = set of retail outlets = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$
- $x_{ij} \geq 0$ = quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous)
- $d_j$ = demand at outlet $j$ (from customer_demand.csv)
- $s_i$ = supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (plants): All unique values in column supplier_id of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (outlets): All unique values in column customer_id of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: For each $j \in J$, demand from column demand in table_id file_0_view_0 where customer_id = $j$
- $s_i$: For each $i \in I$, supply_capacity from column supply_capacity in table_id file_1_view_0 where supplier_id = $i$
- $c_{ij}$: For each $i \in I$, $j \in J$, value from table_id file_2_view_0, row where supplier_id = $i$, column transportation_cost_to_$j$ (where $j$ is the customer_id, e.g., C1, C2, etc.)

Index sets, parameters, and all coefficients are to be taken exactly as specified in the current Observation, preserving all identifiers and source order.