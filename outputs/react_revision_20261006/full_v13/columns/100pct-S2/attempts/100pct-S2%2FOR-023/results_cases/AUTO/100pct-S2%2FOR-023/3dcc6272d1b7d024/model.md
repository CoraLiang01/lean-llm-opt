##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers (facility locations), indexed by $i$.
- $J$: Set of stores (customers), indexed by $j$.

##### Parameters

- $d_j$: Demand of store $j$.
- $f_i$: Fixed cost to activate supplier $i$.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ (a valid upper bound for each supplier, since there are no explicit supplier capacity limits).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All unique values in column `"Unnamed: 2"` of `fixed_cost.csv` (`table_id`: file_1_view_0).
- $J$: All unique values in column `"Customer"` of `demand.csv` (`table_id`: file_0_view_0).
- $d_j$: Value in column `"demand"` for each $j$ in `demand.csv` (`table_id`: file_0_view_0).
- $f_i$: Value in column `"fixed_costs"` for each $i$ in `fixed_cost.csv` (`table_id`: file_1_view_0).
- $c_{ij}$: Value in column with header matching store $j$ for row with `"Unnamed: 2" = i` in `transportation_costs.csv` (`table_id`: file_2_view_0`).
- $M_i$: $\sum_{j \in J} d_j$ (sum over all $d_j$ from `demand.csv`).

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.