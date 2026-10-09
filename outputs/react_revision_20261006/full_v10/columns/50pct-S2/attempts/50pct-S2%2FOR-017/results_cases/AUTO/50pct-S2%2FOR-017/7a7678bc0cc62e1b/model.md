Mathematical Model

Sets:
- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$ (from file_1_view_0.supplier_id)
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$ (from file_0_view_0.customer_id)

Parameters:
- $d_j$: demand of customer $j \in J$ (from file_0_view_0.demand)
- $s_i$: supply capacity of supplier $i \in I$ (from file_1_view_0.supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" = $i$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer group:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity for each supplier:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (suppliers): file_1_view_0.supplier_id
- $J$ (customers): file_0_view_0.customer_id
- $d_j$: file_0_view_0, column "demand", row "customer_id" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row "supplier_id" = $i$
- $c_{ij}$: file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" = $i$
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$