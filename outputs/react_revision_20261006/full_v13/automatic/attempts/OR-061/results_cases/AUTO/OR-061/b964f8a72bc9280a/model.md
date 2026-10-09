##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   x_{ij} \leq U_{ij} y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $U_{ij}$ is any valid upper bound on $x_{ij}$ (e.g., $U_{ij} = d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of suppliers, from column `Unnamed: 0` in `file_1_view_0` and `file_2_view_0`.
- $J$: Set of branches, from column `customer` in `file_0_view_0` and columns in `file_2_view_0` (excluding `Unnamed: 0`).
- $d_j$: Demand of branch $j$, from column `demand` in `file_0_view_0`.
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from matrix in `file_2_view_0` (row: supplier $i$ from `Unnamed: 0`, column: branch $j$).
- $U_{ij}$: Upper bound for $x_{ij}$, set to $d_j$ (from `file_0_view_0`) for each $j$.

##### Data Mapping

- $I$: All values in `file_1_view_0` column `Unnamed: 0` and `file_2_view_0` rows `Unnamed: 0`.
- $J$: All values in `file_0_view_0` column `customer` and `file_2_view_0` columns (excluding `Unnamed: 0`).
- $d_j$: `file_0_view_0` column `demand`, indexed by `customer`.
- $f_i$: `file_1_view_0` column `fixed_costs`, indexed by `Unnamed: 0`.
- $c_{ij}$: `file_2_view_0` matrix, rows indexed by `Unnamed: 0` (supplier), columns by branch label.
- $U_{ij}$: $d_j$ from `file_0_view_0` for each $j$.

No supplier capacity limits are specified, so only demand and activation logic are enforced.