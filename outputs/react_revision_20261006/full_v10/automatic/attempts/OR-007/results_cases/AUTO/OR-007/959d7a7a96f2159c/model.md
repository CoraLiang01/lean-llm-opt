Mathematical Model

Sets:
- $I$: set of warehouses (indexed by $i$), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$: set of stores (indexed by $j$), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand of store $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of warehouse $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity for each warehouse:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (warehouses): file_1_view_0, column "region"
- $J$ (stores): file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "region"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (where $i$ matches "region" and $j$ matches "customer")