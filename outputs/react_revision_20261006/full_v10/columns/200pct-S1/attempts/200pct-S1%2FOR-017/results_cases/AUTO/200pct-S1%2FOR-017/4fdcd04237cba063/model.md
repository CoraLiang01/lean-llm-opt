Mathematical Model

Sets:
- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$ (from file_1_view_0, column supplier_id)
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$ (from file_0_view_0, column customer_id)

Parameters:
- $d_j$: demand of customer $j \in J$ (from file_0_view_0, column demand)
- $s_i$: supply capacity of supplier $i \in I$ (from file_1_view_0, column supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column transportation_cost_to_Ck for each $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous)

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

- $I$ (suppliers): file_1_view_0, column supplier_id
- $J$ (customers): file_0_view_0, column customer_id
- $d_j$: file_0_view_0, column demand, row with customer_id $j$
- $s_i$: file_1_view_0, column supply_capacity, row with supplier_id $i$
- $c_{ij}$: file_2_view_0, row with supplier_id $i$, column transportation_cost_to_$j$ (where $j$ is the customer_id, e.g., transportation_cost_to_C1 for $j$ = C1)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$