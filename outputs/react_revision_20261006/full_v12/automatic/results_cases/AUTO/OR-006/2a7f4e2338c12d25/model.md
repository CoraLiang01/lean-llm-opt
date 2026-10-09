#### Mathematical Model

Let $I$ be the set of warehouses (from supply_capacity.csv, column "Unnamed: 0"), and $J$ the set of stores (from customer_demand.csv, column "customer").

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the unit transportation cost from warehouse $i$ to store $j$ (from transportation_costs.csv, table_id file_2_view_0, columns $J$, rows $I$).

Subject to:
1. Demand satisfaction for each store:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand of store $j$ (from customer_demand.csv, column "demand", table_id file_0_view_0).

2. Supply capacity for each warehouse:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of warehouse $i$ (from supply_capacity.csv, column "supply_capacity", table_id file_1_view_0).

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: warehouse IDs from supply_capacity.csv, column "Unnamed: 0", table_id file_1_view_0.
- $J$: store IDs from customer_demand.csv, column "customer", table_id file_0_view_0.
- $d_j$: demand for store $j$ from customer_demand.csv, column "demand", table_id file_0_view_0.
- $s_i$: supply capacity for warehouse $i$ from supply_capacity.csv, column "supply_capacity", table_id file_1_view_0.
- $c_{ij}$: transportation cost from warehouse $i$ to store $j$ from transportation_costs.csv, table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

All sets, parameters, and coefficients are to be taken exactly as listed in the current CSV files, preserving all identifiers and source order.