#### Mathematical Model

Let:
- $I$ = set of production plants = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J$ = set of retail outlets = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$
- $x_{ij} \geq 0$ = quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous)
- $d_j$ = demand of outlet $j \in J$
- $s_i$ = supply capacity of plant $i \in I$
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$

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

- $I$ (plants): all values in column "Unnamed: 0" of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (outlets): all values in column "customer" of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: column "demand" in file_0_view_0, indexed by "customer"
- $s_i$: column "supply_capacity" in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: value in file_2_view_0 (transportation_costs.csv), row "Unnamed: 0" = $i$, column $j$ (C1, C2, C3, C4)
- $x_{ij}$: decision variable for each $i \in I$, $j \in J$