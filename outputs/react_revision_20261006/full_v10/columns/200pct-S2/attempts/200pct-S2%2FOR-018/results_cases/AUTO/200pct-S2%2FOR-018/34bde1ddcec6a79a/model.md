Mathematical Model

Sets:
- $I$: set of distribution centers (suppliers), $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

Parameters:
- $d_j$: demand of customer group $j \in J$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of distribution center $i \in I$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_$j$" for row $i$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous)

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

- $I$ (distribution centers): all "supplier_id" in file_1_view_0 and file_2_view_0, in source order.
- $J$ (customer groups): all "customer_id" in file_0_view_0 and file_2_view_0, in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id" $j$.
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id" $i$.
- $c_{ij}$: file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" $i$; $j$ matches the customer group suffix in the column name.

All sets, parameters, and indices are defined by the current source data, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.