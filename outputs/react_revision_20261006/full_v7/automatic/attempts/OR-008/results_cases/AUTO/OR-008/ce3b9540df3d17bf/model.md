##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Mathematical Model

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction (each store's demand is met):**
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$

2. **Supply capacity (each warehouse's shipments do not exceed its capacity):**
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): "Suppliers" column in supply_capacity.csv and "Unnamed: 0" in transportation_costs.csv, in source order: Supplier1, Supplier2, Supplier3, Supplier4, Supplier5.
- $J$ (stores): "Customers" column in customer_demand.csv and columns Customer1–Customer6 in transportation_costs.csv, in source order: Customer1, Customer2, Customer3, Customer4, Customer5, Customer6.
- $d_j$: demand for each $j$ from customer_demand.csv, column "demand".
- $s_i$: supply capacity for each $i$ from supply_capacity.csv, column "supply_capacity".
- $c_{ij}$: transportation cost from transportation_costs.csv, with rows indexed by "Unnamed: 0" (warehouses) and columns by Customer1–Customer6 (stores), in source order.

All indices, parameters, and coefficients are to be used exactly as retrieved from the source files.