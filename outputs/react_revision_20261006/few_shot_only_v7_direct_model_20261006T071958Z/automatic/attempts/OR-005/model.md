##### Mathematical Model

Let $I$ be the set of suppliers (distribution centers) and $J$ the set of customers (customer groups):

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Define variables:
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

Subject to:

1. **Demand satisfaction** (each customer receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand for customer $j$.

2. **Supply capacity** (each supplier does not exceed its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity**:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (suppliers): all "Supplier" values from table_id="file_1_view_0", column "Supplier"
- $J$ (customers): all "Customers" values from table_id="file_0_view_0", column "Customers"
- $d_j$: demand for customer $j$ from table_id="file_0_view_0", column "demand"
- $s_i$: supply capacity for supplier $i$ from table_id="file_1_view_0", column "supply_capacity"
- $c_{ij}$: transportation cost from supplier $i$ to customer $j$ from table_id="file_2_view_0", row where "Unnamed: 0" = $i$, column $j$