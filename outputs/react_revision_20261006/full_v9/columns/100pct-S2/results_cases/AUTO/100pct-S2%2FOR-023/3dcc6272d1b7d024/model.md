##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large upper bound for each supplier (since no explicit supplier capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility names from `file_1_view_0` column `Unnamed: 2`)
- $J$: Set of stores (customer names from `file_0_view_0` column `Customer`)
- $d_j$: Demand for store $j$ (from `file_0_view_0` column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0` column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 2` = $i$, column header = $j$)
- $M_i$: Big-M upper bound for supplier $i$ (set to $\sum_{j \in J} d_j$)

##### Data Mapping

- $I$: All unique values in `file_1_view_0` column `Unnamed: 2`
- $J$: All unique values in `file_0_view_0` column `Customer`
- $d_j$: `file_0_view_0` column `demand`, indexed by `Customer`
- $f_i$: `file_1_view_0` column `fixed_costs`, indexed by `Unnamed: 2`
- $c_{ij}$: `file_2_view_0` matrix, rows indexed by `Unnamed: 2` (supplier), columns by store names (column headers matching $J$)
- $M_i$: $\sum_{j \in J} d_j$ (sum over all `file_0_view_0` column `demand`)

All index sets and parameters are defined exactly as present in the source data. No additional constraints or bounds are imposed beyond those described above.