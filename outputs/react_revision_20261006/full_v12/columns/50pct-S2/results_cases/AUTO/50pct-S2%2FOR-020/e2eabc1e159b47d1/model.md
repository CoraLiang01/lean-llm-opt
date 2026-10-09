#### Mathematical Model

Let $I$ be the set of warehouses (from supplier_id in supply_capacity.csv and transportation_costs.csv): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$.

Let $J$ be the set of stores (from customer_id in customer_demand.csv and transportation_costs.csv): $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$.

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j$ (from file_0_view_0, column demand_units)
- $s_i$: supply capacity of warehouse $i$ (from file_1_view_0, column supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns transportation_cost_to_D1, ..., transportation_cost_to_D5)

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

- $I$ (warehouses): supplier_id in file_1_view_0 and file_2_view_0
- $J$ (stores): customer_id in file_0_view_0 and columns transportation_cost_to_D* in file_2_view_0
- $d_j$: file_0_view_0, column demand_units, row with customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity_units, row with supplier_id $i$
- $c_{ij}$: file_2_view_0, row with supplier_id $i$, column transportation_cost_to_$j$ (where $j$ matches D1, D2, D3, D4, D5)

All index sets, parameters, and constraints are mapped directly to the current source data as described above.