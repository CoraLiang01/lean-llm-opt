Mathematical Model

Sets:
- $I$: set of suppliers (stores), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

Parameters:
- $d_j$: demand of customer $j \in J$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i \in I$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

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

- $I$ (suppliers): All values in file_1_view_0, column "Unnamed: 0"
- $J$ (customers): All values in file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (column names match customer IDs)

Variable domain and all constraints are as specified in the user query and mapped to the current data. No data is omitted or aggregated.