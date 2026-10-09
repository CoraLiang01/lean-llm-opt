#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of retail stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Parameters:
- $d_j$: demand of store $j \in J$
- $s_i$: supply capacity of warehouse $i \in I$
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity for each warehouse:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (warehouses): All values in column "Unnamed: 0" of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (stores): All values in column "customer" of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: "demand" column in file_0_view_0, mapped by "customer"
- $s_i$: "supply_capacity" column in file_1_view_0, mapped by "Unnamed: 0"
- $c_{ij}$: Entry in file_2_view_0 (transportation_costs.csv) at row with "Unnamed: 0" = $i$, column $j$ (column names "C1", ..., "C10")

All index sets, parameters, and constraints are defined exactly as in the current source data. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source files.