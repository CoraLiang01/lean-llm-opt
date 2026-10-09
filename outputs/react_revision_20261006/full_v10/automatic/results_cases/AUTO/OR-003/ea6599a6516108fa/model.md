Mathematical Model

Sets:
- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (from file_1_view_0, column "Unnamed: 0")
- $J$: set of customer groups, $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (from file_0_view_0, column "customer")

Parameters:
- $d_j$: demand of customer group $j \in J$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i \in I$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer group $j \in J$ (continuous)

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

- $I$ (suppliers): file_1_view_0, column "Unnamed: 0"
- $J$ (customers): file_0_view_0, column "customer"
- $d_j$: file_0_view_0, columns "customer", "demand"
- $s_i$: file_1_view_0, columns "Unnamed: 0", "supply_capacity"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (columns "C1"..."C10")