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

- $I$: Set of suppliers, from column `Unnamed: 0` in `file_1_view_0` (`fixed_cost.csv`)
- $J$: Set of supermarkets, from column `customer` in `file_0_view_0` (`demand.csv`)
- $d_j$: Demand of supermarket $j$, from column `demand` in `file_0_view_0` (`demand.csv`)
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from row `Unnamed: 0` = $i$, column $j$ in `file_2_view_0` (`transportation_costs.csv`)

##### Data Mapping

- $I$ = all values in `file_1_view_0`.`Unnamed: 0`
- $J$ = all values in `file_0_view_0`.`customer`
- $d_j$ = `file_0_view_0`.`demand` for $j$
- $f_i$ = `file_1_view_0`.`fixed_costs` for $i$
- $c_{ij}$ = `file_2_view_0` row `Unnamed: 0` = $i$, column $j$
- $x_{ij}$, $y_i$ as defined above

All parameters and sets are to be taken directly from the referenced columns and rows in the current CSV files.