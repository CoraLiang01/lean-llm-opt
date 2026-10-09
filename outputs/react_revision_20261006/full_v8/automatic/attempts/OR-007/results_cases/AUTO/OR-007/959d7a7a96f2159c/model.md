#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: demand of store $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of warehouse $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$ (from file_2_view_0, columns "D1"–"D5", rows indexed by "Unnamed: 0")

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Store demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Warehouse supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (warehouses): all "region" values in file_1_view_0 (supply_capacity.csv), in source order.
- $J$ (stores): all "customer" values in file_0_view_0 (customer_demand.csv), in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "region".
- $c_{ij}$: file_2_view_0, columns "D1"–"D5", rows indexed by "Unnamed: 0" (warehouse), columns indexed by store "D1"–"D5".

All indices, parameters, and constraints are mapped directly to the current source data as described above.