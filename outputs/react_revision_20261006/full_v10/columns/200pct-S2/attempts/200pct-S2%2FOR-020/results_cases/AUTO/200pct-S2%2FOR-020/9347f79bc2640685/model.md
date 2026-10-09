Mathematical Model

Sets:
- $S$: set of warehouses (indexed by $i$), from file_1_view_0.supplier_id and file_2_view_0.supplier_id
- $D$: set of stores (indexed by $j$), from file_0_view_0.customer_id

Parameters:
- $d_j$: demand (units) for store $j\in D$, from file_0_view_0.demand_units
- $s_i$: supply capacity (units) for warehouse $i\in S$, from file_1_view_0.supply_capacity_units
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from file_2_view_0, columns transportation_cost_to_Dk

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
$$
\min \sum_{i\in S} \sum_{j\in D} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction for each store:
$$
\sum_{i\in S} x_{ij} \geq d_j \quad \forall j\in D
$$

2. Supply capacity for each warehouse:
$$
\sum_{j\in D} x_{ij} \leq s_i \quad \forall i\in S
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i\in S,\, j\in D
$$

Data Mapping

Sets:
- $S$ (warehouses): file_1_view_0.supplier_id and file_2_view_0.supplier_id (S1, S2, S3, S4, S5)
- $D$ (stores): file_0_view_0.customer_id (D1, D2, D3, D4, D5)

Parameters:
- $d_j$: file_0_view_0.demand_units, indexed by customer_id
- $s_i$: file_1_view_0.supply_capacity_units, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_Dk for store $j$ (D1–D5)

Variables:
- $x_{ij}$: quantity shipped from warehouse $i$ (file_1_view_0.supplier_id) to store $j$ (file_0_view_0.customer_id), continuous, $\geq 0$

All indices, parameters, and constraints are mapped directly to the current CSV data as described above.