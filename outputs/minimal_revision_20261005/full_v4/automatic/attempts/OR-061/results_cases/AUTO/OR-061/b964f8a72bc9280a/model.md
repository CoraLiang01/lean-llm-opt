##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from `file_1_view_0.Unnamed: 0`
- $J$: Set of branches, from `file_0_view_0.customer`
- $d_j$: Demand of branch $j$, from `file_0_view_0.demand`
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0.fixed_costs`
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from `file_2_view_0` (row: `Unnamed: 0` = $i$, column: $j$)
- $M$: A sufficiently large constant, $M = \sum_{j \in J} d_j$ (total demand)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction (each branch must receive its demand):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (inactive suppliers cannot ship):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All supplier IDs from `file_1_view_0.Unnamed: 0`
- $J$: All branch IDs from `file_0_view_0.customer`
- $d_j$: `file_0_view_0.demand` where `customer` = $j$
- $f_i$: `file_1_view_0.fixed_costs` where `Unnamed: 0` = $i$
- $c_{ij}$: `file_2_view_0` value at row `Unnamed: 0` = $i$, column $j$
- $M$: $\sum_{j \in J} d_j$ (sum over all `file_0_view_0.demand`)

All index sets and parameters are defined directly from the CSV data as described above.