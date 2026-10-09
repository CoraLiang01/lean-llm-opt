##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each customer receives at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each supplier ships no more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (Suppliers): from supply_capacity.csv, column "Suppliers", table_id: file_1_view_0
- $J$ (Customers): from customer_demand.csv, column "Customers", table_id: file_0_view_0
- $d_j$: from customer_demand.csv, column "demand", table_id: file_0_view_0, indexed by "Customers"
- $s_i$: from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0, indexed by "Suppliers"
- $c_{ij}$: from transportation_costs.csv, table_id: file_2_view_0, rows indexed by "Unnamed: 0" (Suppliers), columns indexed by customer names (Customers)

All indices, parameters, and coefficients are mapped directly from the provided CSV files as described above.