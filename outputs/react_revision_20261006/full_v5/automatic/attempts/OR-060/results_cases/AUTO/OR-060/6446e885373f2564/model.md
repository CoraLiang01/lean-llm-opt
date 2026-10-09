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
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), since there are no explicit supplier capacity limits.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" and row axis of "file_2_view_0".
- $J$: set of supermarkets, from column "customer" in table_id "file_0_view_0" and column axis of "file_2_view_0".
- $d_j$: demand of supermarket $j$, from column "demand" in table_id "file_0_view_0".
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0".
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from table_id "file_2_view_0", row "Unnamed: 0" (supplier), column $J$ (supermarket).
- $M = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All values in "Unnamed: 0" of "file_1_view_0" and row axis of "file_2_view_0".
- $J$: All values in "customer" of "file_0_view_0" and column axis of "file_2_view_0".
- $d_j$: "demand" column in "file_0_view_0", indexed by "customer".
- $f_i$: "fixed_costs" column in "file_1_view_0", indexed by "Unnamed: 0".
- $c_{ij}$: "file_2_view_0", with rows indexed by "Unnamed: 0" (supplier), columns by supermarket IDs $J$.
- $M$: $\sum_{j \in J} d_j$, using all $d_j$ from "file_0_view_0".

No supplier capacity limits are present; $M$ is a valid upper bound for activation constraints. All index sets and parameters are defined directly from the CSV data.