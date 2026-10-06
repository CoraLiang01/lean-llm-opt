##### Symbolic Model

Let $I$ be the set of warehouses (suppliers) and $J$ the set of stores (customers):

$I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$

$J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

Let $x_{ij} \geq 0$ be the amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $d_j$ (store demand): from customer_demand.csv, table_id: file_0_view_0, columns: Customers, demand, rows 0–5
- $s_i$ (warehouse supply capacity): from supply_capacity.csv, table_id: file_1_view_0, columns: Suppliers, supply_capacity, rows 0–4
- $c_{ij}$ (transportation cost): from transportation_costs.csv, table_id: file_2_view_0, columns: Unnamed: 0 (supplier), Customer1–Customer6, rows 0–4

All identifiers and coefficients are to be used exactly as in the retrieved tables.