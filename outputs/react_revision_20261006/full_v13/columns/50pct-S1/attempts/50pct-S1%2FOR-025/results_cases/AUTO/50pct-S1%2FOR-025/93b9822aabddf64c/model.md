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

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" (fixed_cost.csv)
- $J$: Set of supermarkets, from column "customer" in table_id "file_0_view_0" (demand.csv)
- $d_j$: Demand of supermarket $j$, from column "demand" in table_id "file_0_view_0" (demand.csv)
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0" (fixed_cost.csv)
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from table_id "file_2_view_0" (transportation_costs.csv), with rows indexed by supplier "Unnamed: 0" and columns by supermarket "customer"

##### Data Mapping

- $I = \{$ all values in "Unnamed: 0" of "file_1_view_0" $\}$
- $J = \{$ all values in "customer" of "file_0_view_0" $\}$
- $d_j$: "demand" column in "file_0_view_0", indexed by "customer"
- $f_i$: "fixed_costs" column in "file_1_view_0", indexed by "Unnamed: 0"
- $c_{ij}$: "file_2_view_0" matrix, rows "Unnamed: 0" (suppliers), columns "C1", "C2" (supermarkets)

All parameters and sets are defined directly from the current CSV data as described above.