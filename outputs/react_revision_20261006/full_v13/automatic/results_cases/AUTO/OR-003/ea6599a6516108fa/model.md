Mathematical Model

Sets:
- $I$: set of suppliers, indexed by $i$ (from file_1_view_0, column "Unnamed: 0")
- $J$: set of customer groups, indexed by $j$ (from file_0_view_0, column "customer")

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

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

- $I$ (suppliers): file_1_view_0, column "Unnamed: 0"
- $J$ (customer groups): file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", row where "customer" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row where "Unnamed: 0" = $i$
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (column names match "customer" in file_0_view_0)

Variable domain and all constraints are as specified in the user query and mapped to the current data. No data is omitted or aggregated.