##### Decision Variables

For each supplier $i \in I$ and customer $j \in J$, let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand for customer $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, row "Unnamed: 0" = supply*, column = demand*)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each supplier $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (suppliers): file_1_view_0, column "Supplier"
- $J$ (customers): file_0_view_0, column "Customers"
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Supplier"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), column (customer demand column, e.g., "demand1", ..., "demand8")