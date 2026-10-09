##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv).
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): All supplier_id values in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All customer_id values in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand column in file_0_view_0, indexed by customer_id.
- $s_i$: supply_capacity column in file_1_view_0, indexed by supplier_id.
- $c_{ij}$: transportation_cost_to_Ck columns in file_2_view_0 (transportation_costs.csv), with $i$ = supplier_id (row), $j$ = customer_id (column), using the relationships mapping in the Observation.

All index sets, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and source order.