##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in I,\, j \in J
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: set of suppliers, from `file_1_view_0`, column `Unnamed: 0`
- $J$: set of supermarkets, from `file_0_view_0`, column `customer`
- $d_j$: demand of supermarket $j$, from `file_0_view_0`, column `demand`
- $f_i$: fixed cost for supplier $i$, from `file_1_view_0`, column `fixed_costs`
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$, from `file_2_view_0`, row `Unnamed: 0` (supplier), columns `C1`, `C2` (supermarkets)

##### Data Mapping

- $I$ = all values in `file_1_view_0`, column `Unnamed: 0`
- $J$ = all values in `file_0_view_0`, column `customer`
- $d_j$ = `file_0_view_0`, column `demand`, indexed by $j$
- $f_i$ = `file_1_view_0`, column `fixed_costs`, indexed by $i$
- $c_{ij}$ = `file_2_view_0`, row `Unnamed: 0` (supplier $i$), column $j$ (supermarket $j$)

All parameters and sets are defined directly from the CSV sources as described above.