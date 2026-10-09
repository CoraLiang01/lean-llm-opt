##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand at store $j$ (from customer_demand.csv)
- $s_i$: supply capacity at warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (warehouses): All unique values in column supplier_id of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (stores): All unique values in column customer_id of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: demand_units from table_id file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity_units from table_id file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Dk columns from table_id file_2_view_0 (transportation_costs.csv), with row index supplier_id and column index Dk (where Dk matches customer_id in J)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ as defined above

All index sets, parameters, and constraints are mapped directly to the current CSV data as described.