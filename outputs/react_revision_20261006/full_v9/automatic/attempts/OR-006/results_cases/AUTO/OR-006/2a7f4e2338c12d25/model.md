Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of retail stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Parameters:
- $d_j$: demand of store $j$ (from file_0_view_0, column "demand", indexed by "customer")
- $s_i$: supply capacity of warehouse $i$ (from file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0")
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity for each warehouse:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

Data Mapping

- $I$ (warehouses): file_1_view_0, column "Unnamed: 0", all rows
- $J$ (stores): file_0_view_0, column "customer", all rows
- $d_j$: file_0_view_0, columns "customer" (key), "demand" (value)
- $s_i$: file_1_view_0, columns "Unnamed: 0" (key), "supply_capacity" (value)
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (where $j$ matches "customer" in file_0_view_0)

All index sets, parameters, and constraints are defined exactly as in the current data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.