Mathematical Model

Let $I$ be the set of production plants (indexed by $i$), and $J$ the set of retail outlets (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ from file_1_view_0.Unnamed: 0
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ from file_0_view_0.customer

Parameters:
- $d_j$: daily demand at outlet $j$ (file_0_view_0.demand)
- $s_i$: daily production capacity at plant $i$ (file_1_view_0.supply_capacity)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (file_2_view_0, row $i$, column $j$)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous)

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

Data Mapping

- $I$ (plants): file_1_view_0.Unnamed: 0 = {"S1", "S2", "S3", "S4"}
- $J$ (retail outlets): file_0_view_0.customer = {"C1", "C2", "C3", "C4"}
- $d_j$: file_0_view_0.demand, indexed by file_0_view_0.customer
- $s_i$: file_1_view_0.supply_capacity, indexed by file_1_view_0.Unnamed: 0
- $c_{ij}$: file_2_view_0, row Unnamed: 0 = $i$, column = $j$ (C1, C2, C3, C4)

Variable domain: $x_{ij} \geq 0$, continuous, for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are mapped directly to the current source data as specified above.