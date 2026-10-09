Mathematical Model

Sets:
- $I$: set of suppliers, indexed by $i$ (from supplier_id in file_1_view_0 and file_2_view_0)
- $J$: set of customer groups, indexed by $j$ (from customer_id in file_0_view_0 and file_2_view_0)

Parameters:
- $d_j$: demand of customer group $j$ (from column demand in file_0_view_0)
- $s_i$: supply capacity of supplier $i$ (from column supply_capacity in file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_Ck for customer $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer group $j$ (continuous)

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

- $I$: All supplier_id in file_1_view_0 and file_2_view_0 (S1, S2, ..., S10)
- $J$: All customer_id in file_0_view_0 and columns transportation_cost_to_Ck in file_2_view_0 (C1, C2, ..., C10)
- $d_j$: demand from file_0_view_0, column demand, for customer_id $j$
- $s_i$: supply_capacity from file_1_view_0, column supply_capacity, for supplier_id $i$
- $c_{ij}$: transportation_cost_to_Ck from file_2_view_0, row with supplier_id $i$, column for customer $j$ (e.g., transportation_cost_to_C3 for $j$ = C3)
- $x_{ij}$: decision variable for shipment from $i$ to $j$ (continuous, nonnegative)

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order.