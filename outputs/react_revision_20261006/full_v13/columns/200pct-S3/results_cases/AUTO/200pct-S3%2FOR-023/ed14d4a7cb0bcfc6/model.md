##### Decision Variables

- $x_{ij} \geq 0$: quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

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
   where $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 3" in `fixed_cost.csv` and "Unnamed: 4" in `transportation_costs.csv`.
- $J$: set of stores, from column "Customer" in `demand.csv` and columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT" in `transportation_costs.csv`.
- $d_j$: demand for store $j$, from column "demand" in `demand.csv`.
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$, from the matrix in `transportation_costs.csv` (row "Unnamed: 4" = supplier $i$, column = store $j$).
- $M_i$: upper bound for supplier $i$ (may use $M_i = \sum_{j \in J} d_j$).

##### Data Mapping

- $I$: All unique values in `fixed_cost.csv` column "Unnamed: 3" and `transportation_costs.csv` column "Unnamed: 4".  
  (table_id: file_1_view_0, column: "Unnamed: 3"; table_id: file_2_view_0, column: "Unnamed: 4")
- $J$: All unique values in `demand.csv` column "Customer" and `transportation_costs.csv` columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"].  
  (table_id: file_0_view_0, column: "Customer"; table_id: file_2_view_0, columns: ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"])
- $d_j$: (table_id: file_0_view_0, columns: "Customer", "demand")
- $f_i$: (table_id: file_1_view_0, columns: "Unnamed: 3", "fixed_costs")
- $c_{ij}$: (table_id: file_2_view_0, row: "Unnamed: 4" = supplier $i$, columns: store $j$)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (derived from all $d_j$ above)

All parameters and index sets are to be taken directly from the referenced columns and rows in the source CSVs.