##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all suppliers $i$ and customers $j$.

##### Sets

- $I$: set of suppliers, as listed in column "Unnamed: 0" of table_id file_1_view_0 and file_2_view_0.
- $J$: set of customers, as listed in column "customer" of table_id file_0_view_0 and columns "C1"–"C10" of file_2_view_0.

##### Parameters

- $d_j$: demand of customer $j$, from column "demand" in file_0_view_0.
- $s_i$: supply capacity of supplier $i$, from column "supply_capacity" in file_1_view_0.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

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

- $I$ (suppliers): all values in "Unnamed: 0" column of file_1_view_0 and file_2_view_0.
- $J$ (customers): all values in "customer" column of file_0_view_0 and columns "C1"–"C10" of file_2_view_0.
- $d_j$: from "demand" column in file_0_view_0, indexed by "customer".
- $s_i$: from "supply_capacity" column in file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: from file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

All indices, parameters, and coefficients are to be taken exactly as listed in the returned tables.