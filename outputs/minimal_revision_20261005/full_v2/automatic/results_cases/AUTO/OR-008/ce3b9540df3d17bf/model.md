##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: Set of warehouses (suppliers), from `supply_capacity.csv`:
  - $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J$: Set of stores (customers), from `customer_demand.csv`:
  - $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- $d_j$: Demand of customer $j$, from `customer_demand.csv` (table_id: file_0_view_0, column: demand)
- $s_i$: Supply capacity of warehouse $i$, from `supply_capacity.csv` (table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$, from `transportation_costs.csv` (table_id: file_2_view_0, row: Unnamed: 0, columns: Customer1–Customer6)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store receives at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each warehouse does not ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses): `supply_capacity.csv` (table_id: file_1_view_0), column "Suppliers"
- $J$ (stores): `customer_demand.csv` (table_id: file_0_view_0), column "Customers"
- $d_j$: `customer_demand.csv` (table_id: file_0_view_0), column "demand", indexed by "Customers"
- $s_i$: `supply_capacity.csv` (table_id: file_1_view_0), column "supply_capacity", indexed by "Suppliers"
- $c_{ij}$: `transportation_costs.csv` (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column = $j$ (Customer1–Customer6)

All indices, coefficients, and constraints are bound directly to the retrieved data. Variable domains and all constraints are as specified in the query.