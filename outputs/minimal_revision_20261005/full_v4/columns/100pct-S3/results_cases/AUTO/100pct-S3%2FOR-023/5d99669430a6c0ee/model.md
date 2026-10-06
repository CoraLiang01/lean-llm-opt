##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 2`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)
- $d_j$: Demand at store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost to activate supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 2` = $i$, column $j$ = store name)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores; a valid upper bound since there are no supplier capacity limits).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All values in `file_1_view_0`, column `Unnamed: 2`
- $J$: All values in `file_0_view_0`, column `Customer`
- $d_j$: `file_0_view_0`, columns: `Customer`, `demand`
- $f_i$: `file_1_view_0`, columns: `Unnamed: 2`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 2` = $i$, column $j$ = store name (column header matches $j$ in $J$)
- $M$: $\sum_{j \in J} d_j$ (sum over all `demand` in `file_0_view_0`)

No additional constraints or capacity limits are imposed beyond those above. All index sets and parameters are defined directly from the CSV data as specified.