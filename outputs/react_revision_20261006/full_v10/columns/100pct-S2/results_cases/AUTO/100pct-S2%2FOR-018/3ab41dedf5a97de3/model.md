Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers), indexed by $i$, with $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$ (from file_1_view_0.supplier_id and file_2_view_0.supplier_id).
Let $J$ be the set of customer groups, indexed by $j$, with $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$ (from file_0_view_0.customer_id and file_2_view_0 column suffixes).

Parameters:
$d_j$: demand of customer group $j$ (from file_0_view_0.demand, indexed by customer_id).
$s_i$: supply capacity of distribution center $i$ (from file_1_view_0.supply_capacity, indexed by supplier_id).
$c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_Ck, row supplier_id).

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

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

- $I$ (distribution centers): file_1_view_0.supplier_id and file_2_view_0.supplier_id (rows)
- $J$ (customer groups): file_0_view_0.customer_id and file_2_view_0 columns with suffix "transportation_cost_to_Ck"
- $d_j$: file_0_view_0.demand, indexed by customer_id
- $s_i$: file_1_view_0.supply_capacity, indexed by supplier_id
- $c_{ij}$: file_2_view_0, value in row with supplier_id $i$ and column "transportation_cost_to_$j$"
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$

All sets, parameters, and indices are defined exactly as in the current source data, preserving all identifiers and their order. No data is omitted or aggregated. All constraints and variable domains are as specified in the user query.