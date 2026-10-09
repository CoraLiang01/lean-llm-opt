##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers) from file_1_view_0: $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$: set of customer groups from file_0_view_0: $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer group $j$ from file_0_view_0, column "demand"
- $s_i$: supply capacity of distribution center $i$ from file_1_view_0, column "supply_capacity"
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ from file_2_view_0, row "Unnamed: 0" (distribution center), columns "demand1"..."demand8" (customer groups)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (each customer group receives at least its demand):
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity (each distribution center does not exceed its capacity):
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (distribution centers): file_1_view_0, column "Supplier"
- $J$ (customer groups): file_0_view_0, column "Customers"
- $d_j$: file_0_view_0, column "demand", indexed by "Customers"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Supplier"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (distribution center), columns "demand1"..."demand8" (customer groups), aligned as per relationships in the Observation

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and mapped as above.