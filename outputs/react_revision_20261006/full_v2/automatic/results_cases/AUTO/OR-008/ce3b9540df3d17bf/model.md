##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse $i$ (supplier) to store $j$ (customer), for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): file_1_view_0, column "Suppliers"
- $J$ (customers): file_0_view_0, column "Customers"
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Suppliers"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = supplier $i$, column = customer $j$