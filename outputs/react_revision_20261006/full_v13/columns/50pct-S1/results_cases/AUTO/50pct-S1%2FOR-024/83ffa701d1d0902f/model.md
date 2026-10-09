##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each warehouse (since no explicit warehouse capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: set of musicians/bands, from column "customer" in table_id file_0_view_0 and columns in file_2_view_0.
- $d_j$: demand of musician/band $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: fixed cost for warehouse $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to musician/band $j$, from table_id file_2_view_0, row "Unnamed: 0" (warehouses), columns $J$.
- $M_i$: big-M upper bound for warehouse $i$, set as $\sum_{j \in J} d_j$.

##### Data Mapping

- $I$: all values in "Unnamed: 0" of file_1_view_0 and file_2_view_0.
- $J$: all values in "customer" of file_0_view_0 and columns (except "Unnamed: 0") of file_2_view_0.
- $d_j$: "demand" in file_0_view_0, indexed by "customer".
- $f_i$: "fixed_costs" in file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: entries in file_2_view_0, rows "Unnamed: 0" (warehouses), columns $J$.
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in file_0_view_0).

All parameters and sets are mapped directly from the current CSV files as described above.