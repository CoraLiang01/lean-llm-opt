Mathematical Model

Sets:
- $I$: set of warehouses (suppliers), $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J$: set of stores (customers), $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

Parameters:
- $d_j$: demand of store $j \in J$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of warehouse $i \in I$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

Decision Variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)

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

- $I$ (warehouses): file_1_view_0, column "Suppliers", all rows
- $J$ (stores): file_0_view_0, column "Customers", all rows
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Suppliers"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$ (warehouse), column $j$ (store, matches "Customers" in file_0_view_0)

All indices, parameters, and coefficients are to be taken exactly as listed in the current CSV files and their source order. No data is omitted or aggregated. Variable domains and all constraints are as specified above.