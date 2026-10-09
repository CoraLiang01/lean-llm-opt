Mathematical Model

Sets:
- $I$: set of distribution centers (sources), indexed by $i$, from file_1_view_0["Unnamed: 0"]
- $J$: set of customer groups (destinations), indexed by $j$, from file_0_view_0["customer"]

Parameters:
- $d_j$: demand of customer group $j$, from file_0_view_0["demand"]
- $s_i$: supply capacity of distribution center $i$, from file_1_view_0["supply_capacity"]
- $c_{ij}$: transportation cost per unit from $i$ to $j$, from file_2_view_0, row $i$ column $j$

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous)

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

- $I$ (distribution centers): file_1_view_0["Unnamed: 0"] (S1, S2, ..., S12)
- $J$ (customer groups): file_0_view_0["customer"] (C1, C2, ..., C12)
- $d_j$: file_0_view_0["demand"], indexed by file_0_view_0["customer"]
- $s_i$: file_1_view_0["supply_capacity"], indexed by file_1_view_0["Unnamed: 0"]
- $c_{ij}$: file_2_view_0, row $i$ = file_2_view_0["Unnamed: 0"], column $j$ = customer group name

Variable domains, objective sense, and all constraints are as specified in the current user query and data. No data is omitted or aggregated.