Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers), indexed by $i$, with $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$ (from file_1_view_0, column supplier_id).
Let $J$ be the set of customer groups, indexed by $j$, with $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$ (from file_0_view_0, column customer_id).

Parameters:
$d_j$: demand (units) for customer group $j \in J$ (from file_0_view_0, column demand_units, key customer_id).
$s_i$: supply capacity (units) for distribution center $i \in I$ (from file_1_view_0, column supply_capacity_units, key supplier_id).
$c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_$j$, row supplier_id $i$).

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each distribution center:
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
- $d_j$: file_0_view_0, column demand_units, key customer_id.
- $s_i$: file_1_view_0, column supply_capacity_units, key supplier_id.
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (column_id_mapping in Observation).

All indices, parameters, and constraints are mapped directly to the current source data, preserving all identifiers and source order. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.