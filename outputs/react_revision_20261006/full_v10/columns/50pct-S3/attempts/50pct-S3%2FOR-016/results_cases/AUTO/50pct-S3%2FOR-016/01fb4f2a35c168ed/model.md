Mathematical Model

Sets:
- $I$: set of distribution centers (suppliers), indexed by $i$, from all supplier_id in file_1_view_0.
- $J$: set of customer groups, indexed by $j$, from all customer_id in file_0_view_0.

Parameters:
- $d_j$: demand (units) for customer group $j$, from demand_units in file_0_view_0.
- $s_i$: supply capacity (units) for distribution center $i$, from supply_capacity_units in file_1_view_0.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from transportation_cost_to_Ck in file_2_view_0, with $i$ = supplier_id and $j$ = Ck.

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand_units for customer_id $j$ in file_0_view_0.
- $s_i$: supply_capacity_units for supplier_id $i$ in file_1_view_0.
- $c_{ij}$: transportation_cost_to_Ck for supplier_id $i$ and customer_id $j$ in file_2_view_0 (transportation_costs.csv), with columns mapped as per Observation relationships.

All indices, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and source order.