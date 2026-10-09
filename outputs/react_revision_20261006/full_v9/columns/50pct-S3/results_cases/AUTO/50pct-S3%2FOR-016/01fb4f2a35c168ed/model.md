#### Abstract Symbolic Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$ (from all supplier_id in supply_capacity.csv)
- $J$ = set of customer groups, indexed by $j$ (from all customer_id in customer_demand.csv)
- $d_j$ = demand of customer group $j$ (from demand_units in customer_demand.csv)
- $s_i$ = supply capacity of distribution center $i$ (from supply_capacity_units in supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_cost_to_C* columns in transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous)

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

- $I$ (distribution centers): All supplier_id in supply_capacity.csv (table_id: file_1_view_0, column: supplier_id)
- $J$ (customer groups): All customer_id in customer_demand.csv (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand_units for customer_id $j$ in customer_demand.csv (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$: supply_capacity_units for supplier_id $i$ in supply_capacity.csv (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$: transportation_cost_to_C* for supplier_id $i$ and customer_id $j$ in transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_C* mapped to customer_id by suffix)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$

All index sets, parameters, and constraints are defined exactly as in the current CSV data, preserving all identifiers and source order.