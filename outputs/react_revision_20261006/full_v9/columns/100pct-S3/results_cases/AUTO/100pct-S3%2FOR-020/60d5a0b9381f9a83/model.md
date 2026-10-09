Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand (units) for store $j$ (from file_0_view_0, column demand_units)
- $s_i$: supply capacity (units) for warehouse $i$ (from file_1_view_0, column supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns transportation_cost_to_D1, ..., transportation_cost_to_D5)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

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

Data Mapping

- $I$ (warehouses): file_1_view_0, column supplier_id, all rows
- $J$ (stores): file_0_view_0, column customer_id, all rows
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity_units, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (where $j$ is D1, D2, D3, D4, D5)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$