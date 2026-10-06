##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from column `Unnamed: 0` in `file_1_view_0` and `file_2_view_0`.
- $J$: Set of supermarkets, from column `customer` in `file_0_view_0` and columns `C1`, `C2` in `file_2_view_0`.
- $d_j$: Demand of supermarket $j$, from column `demand` in `file_0_view_0`.
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0`.
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from `file_2_view_0` (row `Unnamed: 0` = $i$, column $j$).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), ensuring inactive suppliers do not ship goods.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All values in `file_1_view_0`.`Unnamed: 0` and `file_2_view_0`.`Unnamed: 0`
- $J$: All values in `file_0_view_0`.`customer` and `file_2_view_0` columns `C1`, `C2`
- $d_j$: `file_0_view_0` table, column `demand`, indexed by `customer`
- $f_i$: `file_1_view_0` table, column `fixed_costs`, indexed by `Unnamed: 0`
- $c_{ij}$: `file_2_view_0` table, row `Unnamed: 0` = $i$, column $j$
- $M$: $\sum_{j \in J} d_j$ (sum over all `file_0_view_0`.`demand`)

All sets and parameters are defined exactly as present in the source tables.