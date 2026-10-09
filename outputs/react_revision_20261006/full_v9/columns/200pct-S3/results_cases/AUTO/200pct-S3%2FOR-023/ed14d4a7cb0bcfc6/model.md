##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

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
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large constant (total demand), since no explicit supplier capacity is given.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from `file_1_view_0` column `Unnamed: 3`
- $J$: Set of stores, from `file_0_view_0` column `Customer`
- $d_j$: Demand for store $j$, from `file_0_view_0` column `demand`
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0` column `fixed_costs`
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` row `Unnamed: 4` (supplier) and columns matching store names in $J$
- $M_i$: Big-M parameter for each supplier $i$, set to $\sum_{j \in J} d_j$

##### Data Mapping

- $I$: All unique values in `file_1_view_0` column `Unnamed: 3`
- $J$: All unique values in `file_0_view_0` column `Customer`
- $d_j$: `file_0_view_0` column `demand` for each $j$
- $f_i$: `file_1_view_0` column `fixed_costs` for each $i$
- $c_{ij}$: `file_2_view_0` row where `Unnamed: 4` = $i$, column with header matching $j$ (store name)
- $M_i$: $\sum_{j \in J} d_j$ (computed from all $d_j$)

No additional constraints or capacity limits are imposed unless specified in the data. All index sets and parameters are defined directly from the CSV sources as described.