Mathematical Model

Sets:
- $I$: set of distribution centers (suppliers), indexed by $i$, from file_1_view_0, column supplier_id.
- $J$: set of customer groups, indexed by $j$, from file_0_view_0, column customer_id.

Parameters:
- $d_j$: demand (units) for customer group $j$, from file_0_view_0, column demand_units.
- $s_i$: supply capacity (units) for distribution center $i$, from file_1_view_0, column supply_capacity_units.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from file_2_view_0, column transportation_cost_to_$j$ for row supplier_id $i$.

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column demand_units, for customer_id $j$.
- $s_i$: file_1_view_0, column supply_capacity_units, for supplier_id $i$.
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (where $j$ is the customer_id from file_0_view_0).

All indices, parameters, and coefficients are to be used exactly as returned in the current Observation, preserving source order and identifiers.