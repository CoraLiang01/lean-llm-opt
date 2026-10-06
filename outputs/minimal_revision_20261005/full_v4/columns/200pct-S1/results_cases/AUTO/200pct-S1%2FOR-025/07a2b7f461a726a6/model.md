##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers, from `file_1_view_0.Unnamed: 0`
- $J$: Set of supermarkets, from `file_0_view_0.customer`

##### Parameters and Data Mapping

- $d_j$: Demand of supermarket $j$, from `file_0_view_0.demand`
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0.fixed_costs`
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from `file_2_view_0` (row: `Unnamed: 1` = $i$, column: $j$)
- $M$: $\sum_{j \in J} d_j$ (sum over all $d_j$ from `file_0_view_0.demand`)

##### Data Mapping

- $I$: All values in `file_1_view_0.Unnamed: 0`
- $J$: All values in `file_0_view_0.customer`
- $d_j$: `file_0_view_0.demand` for $j$
- $f_i$: `file_1_view_0.fixed_costs` for $i$
- $c_{ij}$: `file_2_view_0` entry at (row: `Unnamed: 1` = $i$, column: $j$)
- $M$: $\sum_{j \in J} d_j$ (from `file_0_view_0.demand`)

##### Notes

- All index sets and parameters are defined directly from the CSV data as described above.
- No supplier capacity limits are specified beyond the activation logic.