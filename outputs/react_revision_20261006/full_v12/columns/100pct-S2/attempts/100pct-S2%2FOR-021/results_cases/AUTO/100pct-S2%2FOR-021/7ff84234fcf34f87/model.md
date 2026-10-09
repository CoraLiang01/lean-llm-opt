#### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Parameters:
- $d_j$: demand at outlet $j$ (from file_0_view_0, column demand).
- $s_i$: supply capacity at plant $i$ (from file_1_view_0, column supply_capacity).
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, columns transportation_cost_to_C1, ..., _to_C4).

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

- $I$ (plants): supplier_id in file_1_view_0 and file_2_view_0
- $J$ (outlets): customer_id in file_0_view_0 and columns transportation_cost_to_C* in file_2_view_0
- $d_j$: file_0_view_0, column demand, row with customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity, row with supplier_id $i$
- $c_{ij}$: file_2_view_0, row with supplier_id $i$, column transportation_cost_to_$j$
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$