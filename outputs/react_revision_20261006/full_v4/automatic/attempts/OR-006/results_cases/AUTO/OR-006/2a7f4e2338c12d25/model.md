##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of warehouses, from supply_capacity.csv, column "Unnamed: 0", table_id file_1_view_0.
- $J$: set of stores, from customer_demand.csv, column "customer", table_id file_0_view_0.

##### Parameters

- $d_j$: demand of store $j$, from customer_demand.csv, column "demand", table_id file_0_view_0.
- $s_i$: supply capacity of warehouse $i$, from supply_capacity.csv, column "supply_capacity", table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from transportation_costs.csv, table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (each store's demand must be met):
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity (each warehouse's shipments cannot exceed its capacity):
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (warehouses): all values in supply_capacity.csv, column "Unnamed: 0", table_id file_1_view_0.
- $J$ (stores): all values in customer_demand.csv, column "customer", table_id file_0_view_0.
- $d_j$: customer_demand.csv, column "demand", table_id file_0_view_0, for each $j$.
- $s_i$: supply_capacity.csv, column "supply_capacity", table_id file_1_view_0, for each $i$.
- $c_{ij}$: transportation_costs.csv, table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

No data omitted or invented. All indices, parameters, and constraints are mapped to the retrieved source data.