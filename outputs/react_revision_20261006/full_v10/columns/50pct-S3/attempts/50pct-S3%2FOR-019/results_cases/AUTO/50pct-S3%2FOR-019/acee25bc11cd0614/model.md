##### Mathematical Model

Let
- $I$ = set of distribution centers (suppliers): $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$ = set of customer groups (demands): $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous)

Parameters:
- $d_j$: demand of customer $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column "transportation_cost_to_{j}", row "supplier_id" = $i$)

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

- $I$ (suppliers): All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customers): All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", for each $j$.
- $s_i$: file_1_view_0, column "supply_capacity", for each $i$.
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}" (where ${j}$ is the customer_id from file_0_view_0), for each $i, j$.
- $x_{ij}$: decision variable for each $i \in I$, $j \in J$.

All index sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and source order.