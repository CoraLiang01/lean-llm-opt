Mathematical Model

Sets:
- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Parameters:
- $d_j$: demand of customer group $j \in J$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i \in I$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_$j$" for supplier $i$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer group $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer group:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
2. Supply capacity for each supplier:
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (suppliers): All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customers): All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, column "transportation_cost_to_$j$", row "supplier_id" = $i$, column suffix matches $j$ as in file_0_view_0.

All indices, parameters, and coefficients are to be used exactly as returned in the current Observation, preserving source order and identifiers.