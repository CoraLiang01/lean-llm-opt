Mathematical Model

Sets:
- $I$: set of suppliers (from file_1_view_0, column supplier_id)
- $J$: set of customer groups (from file_0_view_0, column customer_id)

Parameters:
- $d_j$: demand of customer $j\in J$ (from file_0_view_0, column demand)
- $s_i$: supply capacity of supplier $i\in I$ (from file_1_view_0, column supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column transportation_cost_to_Ck, row supplier_id)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous)

Objective:
\[
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
\[
\sum_{i\in I} x_{ij} \geq d_j \quad \forall j\in J
\]
2. Supply capacity:
\[
\sum_{j\in J} x_{ij} \leq s_i \quad \forall i\in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i\in I,\, j\in J
\]

Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column demand, indexed by customer_id.
- $s_i$: file_1_view_0, column supply_capacity, indexed by supplier_id.
- $c_{ij}$: file_2_view_0, column transportation_cost_to_Ck (where Ck = customer_id), row supplier_id, indexed by (supplier_id, customer_id).
- $x_{ij}$: decision variable for each (supplier_id, customer_id) pair.

All indices, parameters, and coefficients are to be used exactly as returned in the current Observation, preserving source order and identifiers.