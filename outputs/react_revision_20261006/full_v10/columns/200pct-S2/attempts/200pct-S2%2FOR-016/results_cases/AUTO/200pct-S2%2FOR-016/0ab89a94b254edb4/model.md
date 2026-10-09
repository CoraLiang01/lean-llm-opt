Mathematical Model

Sets:
- $I$: set of distribution centers (suppliers), indexed by $i$, from column "supplier_id" in file_1_view_0 and file_2_view_0.
- $J$: set of customer groups, indexed by $j$, from column "customer_id" in file_0_view_0 and as suffixes in file_2_view_0.

Parameters:
- $d_j$: demand of customer group $j$, from column "demand_units" in file_0_view_0.
- $s_i$: supply capacity of distribution center $i$, from column "supply_capacity_units" in file_1_view_0.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from column "transportation_cost_to_$j$" in file_2_view_0, row "supplier_id" = $i$.

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

- $I$: All "supplier_id" in file_1_view_0 and file_2_view_0, source order.
- $J$: All "customer_id" in file_0_view_0 and as suffixes in file_2_view_0, source order.
- $d_j$: file_0_view_0, column "demand_units", row "customer_id" = $j$.
- $s_i$: file_1_view_0, column "supply_capacity_units", row "supplier_id" = $i$.
- $c_{ij}$: file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" = $i$.
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$.

All indices, parameters, and constraints are defined exactly as in the current source data, preserving source order and identifiers.