Mathematical Model

Sets:
- $I$: set of warehouses (indexed by $i$), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (from file_1_view_0.supplier_id)
- $J$: set of stores (indexed by $j$), $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (from file_0_view_0.customer_id)

Parameters:
- $d_j$: daily demand at store $j$ (from file_0_view_0.demand_units)
- $s_i$: daily supply capacity at warehouse $i$ (from file_1_view_0.supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, column transportation_cost_to_$j$ for row supplier_id $i$)

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

- $I$ (warehouses): file_1_view_0.supplier_id
- $J$ (stores): file_0_view_0.customer_id
- $d_j$: file_0_view_0.demand_units, indexed by customer_id $j$
- $s_i$: file_1_view_0.supply_capacity_units, indexed by supplier_id $i$
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (where $j$ matches customer_id in file_0_view_0)

All indices, parameters, and coefficients are mapped directly from the current CSV sources as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.