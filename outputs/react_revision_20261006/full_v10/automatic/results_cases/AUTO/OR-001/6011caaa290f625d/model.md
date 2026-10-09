Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$):

- $I = \{$S1, S2, ..., S18$\}$ (from supply_capacity.csv, column Unnamed: 0, table_id file_1_view_0)
- $J = \{$C1, C2, ..., C18$\}$ (from customer_demand.csv, column customer, table_id file_0_view_0)

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv, column demand, table_id file_0_view_0)
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv, column supply_capacity, table_id file_1_view_0)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv, entry at row $i$, column $j$, table_id file_2_view_0)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous)

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

- $I$ (distribution centers): all values in column Unnamed: 0 of supply_capacity.csv (table_id file_1_view_0)
- $J$ (customer groups): all values in column customer of customer_demand.csv (table_id file_0_view_0)
- $d_j$: demand for customer $j$ from column demand in customer_demand.csv (table_id file_0_view_0)
- $s_i$: supply capacity for supplier $i$ from column supply_capacity in supply_capacity.csv (table_id file_1_view_0)
- $c_{ij}$: transportation cost from row $i$ (Unnamed: 0) and column $j$ (customer) in transportation_costs.csv (table_id file_2_view_0)
- $x_{ij}$: decision variable for quantity shipped from $i$ to $j$ (continuous, nonnegative)

All index sets, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and bounds.