##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Nonnegativity:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
3. **Supplier activation:**  
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in I,\, j \in J
   \]
   (A supplier can only supply to a supermarket if it is open.)

4. **Binary supplier status:**  
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: set of supermarkets, from column "customer" in table_id file_0_view_0 and columns in file_2_view_0 (excluding "Unnamed: 0").
- $d_j$: demand of supermarket $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

##### Data Mapping

- $I$: All values in "Unnamed: 0" of file_1_view_0 and file_2_view_0.
- $J$: All values in "customer" of file_0_view_0 and all columns (except "Unnamed: 0") in file_2_view_0.
- $d_j$: file_0_view_0, columns "customer", "demand".
- $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs".
- $c_{ij}$: file_2_view_0, rows "Unnamed: 0" = $i$, columns $j$.

All variables and parameters are indexed over the full sets as defined above.