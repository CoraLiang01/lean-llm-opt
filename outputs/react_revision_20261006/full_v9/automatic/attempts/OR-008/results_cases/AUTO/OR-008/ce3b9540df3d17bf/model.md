##### Mathematical Model

Let $I$ be the set of warehouses (suppliers) and $J$ the set of stores (customers):

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from warehouse $i \in I$ to store $j \in J$.

Parameters:
- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

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

##### Data Mapping

- $I$ (warehouses): All "Suppliers" in supply_capacity.csv (table_id: file_1_view_0, column: Suppliers)
- $J$ (stores): All "Customers" in customer_demand.csv (table_id: file_0_view_0, column: Customers)
- $d_j$: "demand" for each $j$ in customer_demand.csv (table_id: file_0_view_0, columns: Customers, demand)
- $s_i$: "supply_capacity" for each $i$ in supply_capacity.csv (table_id: file_1_view_0, columns: Suppliers, supply_capacity)
- $c_{ij}$: entry in transportation_costs.csv (table_id: file_2_view_0, row: Unnamed: 0 = $i$, column: $j$)

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order.