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
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and "Unnamed: 1" in file_2_view_0.
- $J$: set of supermarkets, from column "customer" in table_id file_0_view_0 and columns "C1", "C2" in file_2_view_0.
- $d_j$: demand of supermarket $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$, from table_id file_2_view_0, row "Unnamed: 1" (supplier), columns "C1", "C2" (supermarket).

##### Data Mapping

- $I$: All supplier IDs from file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 1"]
- $J$: All supermarket IDs from file_0_view_0["customer"] and file_2_view_0 columns ["C1", "C2"]
- $d_j$: file_0_view_0["demand"], indexed by file_0_view_0["customer"]
- $f_i$: file_1_view_0["fixed_costs"], indexed by file_1_view_0["Unnamed: 0"]
- $c_{ij}$: file_2_view_0, rows indexed by "Unnamed: 1" (supplier), columns "C1", "C2" (supermarket)

All parameters and sets are defined directly from the current CSV data as described above.