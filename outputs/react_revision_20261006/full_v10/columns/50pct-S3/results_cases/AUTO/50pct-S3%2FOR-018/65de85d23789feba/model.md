Mathematical Model

Sets:
- $I$: set of distribution centers (suppliers), indexed by $i$, from column "supplier_id" in file_1_view_0.
- $J$: set of customer groups, indexed by $j$, from column "customer_id" in file_0_view_0.

Parameters:
- $d_j$: demand of customer group $j$, from column "demand" in file_0_view_0.
- $s_i$: supply capacity of distribution center $i$, from column "supply_capacity" in file_1_view_0.
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from column "transportation_cost_to_$j$" in file_2_view_0, with $i$ from "supplier_id" and $j$ from "customer_id".

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer group:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity for each distribution center:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" = $i$, column suffix $j$ matches "customer_id" in file_0_view_0.

Variable domains, objective sense, and all constraints are as described above, with all index sets and parameters mapped directly from the current source data.