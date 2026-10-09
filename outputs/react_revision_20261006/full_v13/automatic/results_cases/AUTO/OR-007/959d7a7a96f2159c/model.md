#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j \in J$
- $s_i$: supply capacity of warehouse $i \in I$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

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

- $I$ (warehouses): region column of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (stores): customer column of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: demand column of table_id file_0_view_0, indexed by customer
- $s_i$: supply_capacity column of table_id file_1_view_0, indexed by region
- $c_{ij}$: entry in table_id file_2_view_0 (transportation_costs.csv), row indexed by Unnamed: 0 (warehouse/region), column indexed by store/customer

Variable:
- $x_{ij}$: quantity shipped from warehouse $i$ (region in file_1_view_0, Unnamed: 0 in file_2_view_0) to store $j$ (customer in file_0_view_0, column in file_2_view_0), continuous, $\geq 0$.

All index sets, parameters, and constraints are mapped directly to the current source data as described above.