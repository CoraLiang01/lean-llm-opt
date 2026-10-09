##### Mathematical Model

Let $I$ be the set of warehouses (suppliers) and $J$ the set of stores (customers):

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

Let $x_{ij} \geq 0$ be the continuous quantity of fresh produce shipped from warehouse $i \in I$ to store $j \in J$.

Parameters:
- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. **Demand satisfaction** (each store receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. **Supply capacity** (each warehouse does not ship more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. **Non-negativity**:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): values in column "Suppliers" of table_id file_1_view_0 (supply_capacity.csv), source order.
- $J$ (stores): values in column "Customers" of table_id file_0_view_0 (customer_demand.csv), source order.
- $d_j$: column "demand" in table_id file_0_view_0, indexed by "Customers".
- $s_i$: column "supply_capacity" in table_id file_1_view_0, indexed by "Suppliers".
- $c_{ij}$: entry in table_id file_2_view_0 (transportation_costs.csv), row "Unnamed: 0" = $i$, column = $j$.

All indices, coefficients, and constraints are mapped directly from the current CSV data, preserving source order and identifiers.