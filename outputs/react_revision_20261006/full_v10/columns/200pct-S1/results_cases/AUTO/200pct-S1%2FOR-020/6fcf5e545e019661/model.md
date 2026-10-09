Mathematical Model

Sets:
- $I$: set of warehouses (indexed by $i$), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$: set of stores (indexed by $j$), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand (units) for store $j \in J$
- $s_i$: supply capacity (units) for warehouse $i \in I$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

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

- $I$ (warehouses): all unique values in column supplier_id of file_1_view_0 (supply_capacity.csv)
- $J$ (stores): all unique values in column customer_id of file_0_view_0 (customer_demand.csv)
- $d_j$: demand_units from file_0_view_0, mapped by customer_id
- $s_i$: supply_capacity_units from file_1_view_0, mapped by supplier_id
- $c_{ij}$: transportation_cost_to_Dk from file_2_view_0, where $i$ = supplier_id, $j$ = Dk (column suffix), mapped by row supplier_id and column customer_id

- $x_{ij}$: decision variable for each $(i,j)$ pair with $i$ from supplier_id in file_1_view_0 and $j$ from customer_id in file_0_view_0

All indices, parameters, and coefficients are to be used exactly as provided in the current data, preserving their identifiers and source order.