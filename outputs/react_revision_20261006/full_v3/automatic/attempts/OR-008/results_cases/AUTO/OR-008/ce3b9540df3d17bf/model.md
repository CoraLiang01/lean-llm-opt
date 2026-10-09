##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each warehouse cannot ship more than its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $d_j$ is the value in column "demand" for customer $j$ in table_id file_0_view_0 (customer_demand.csv), with $j$ from column "Customers".
- $s_i$ is the value in column "supply_capacity" for supplier $i$ in table_id file_1_view_0 (supply_capacity.csv), with $i$ from column "Suppliers".
- $c_{ij}$ is the value in the matrix at row $i$ (from "Unnamed: 0" in file_2_view_0, transportation_costs.csv) and column $j$ (from "Customer1"..."Customer6" in file_2_view_0).

All indices, parameters, and coefficients are to be taken exactly as listed in the returned tables, preserving their identifiers and order.