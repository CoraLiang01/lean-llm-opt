##### Mathematical Model

Let:
- $I$ = set of warehouses (indexed by $i$), from supplier_id in supply_capacity.csv: $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$ = set of stores (indexed by $j$), from customer_id in customer_demand.csv: $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$
- $d_j$ = demand (units) for store $j$, from demand_units in customer_demand.csv
- $s_i$ = supply capacity (units) for warehouse $i$, from supply_capacity_units in supply_capacity.csv
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to store $j$, from transportation_cost_to_Dk in transportation_costs.csv
- $x_{ij} \geq 0$ = quantity shipped from warehouse $i$ to store $j$ (continuous)

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

- $I$ (warehouses): supplier_id in supply_capacity.csv (table_id: file_1_view_0)
- $J$ (stores): customer_id in customer_demand.csv (table_id: file_0_view_0)
- $d_j$: demand_units in customer_demand.csv (table_id: file_0_view_0, column: demand_units, key: customer_id)
- $s_i$: supply_capacity_units in supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity_units, key: supplier_id)
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, columns: transportation_cost_to_D1 ... transportation_cost_to_D5, mapped to customer_id)
- $x_{ij}$: decision variable, continuous, nonnegative, for all $i \in I$, $j \in J$