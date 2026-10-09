#### Mathematical Model

Let $I$ be the set of warehouses (Suppliers) and $J$ the set of stores (Customers):

$I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$

$J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

Let $x_{ij} \geq 0$ be the amount shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of warehouse $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from file_2_view_0, columns "Customer1"–"Customer6", rows "Supplier1"–"Supplier5")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
- Demand satisfaction:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
- Supply capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
- Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (warehouses): file_1_view_0, column "Suppliers", rows 0–4
- $J$ (stores): file_0_view_0, column "Customers", rows 0–5
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Suppliers"
- $c_{ij}$: file_2_view_0, columns "Customer1"–"Customer6", rows "Supplier1"–"Supplier5", with row IDs from "Unnamed: 0" matching "Suppliers" and column names matching "Customers"

All indices, parameters, and constraints are mapped directly to the current source data.