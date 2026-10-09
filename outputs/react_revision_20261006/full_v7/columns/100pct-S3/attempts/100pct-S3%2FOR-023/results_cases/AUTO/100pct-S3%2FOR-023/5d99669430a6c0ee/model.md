##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

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
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (a valid upper bound on total shipments from any supplier).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 2`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)
- $d_j$: Demand for store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 2` for supplier, column for store)
- $M$: $\sum_{j \in J} d_j$ (sum of all store demands)

##### Data Mapping

- Suppliers $I$: `file_1_view_0`, column `Unnamed: 2`
- Stores $J$: `file_0_view_0`, column `Customer`
- Demand $d_j$: `file_0_view_0`, column `demand`
- Fixed cost $f_i$: `file_1_view_0`, column `fixed_costs`
- Transportation cost $c_{ij}$: `file_2_view_0`, rows indexed by supplier (`Unnamed: 2`), columns indexed by store (column names matching store IDs in $J$)
- $M$: $\sum_{j \in J} d_j$ using `file_0_view_0`, column `demand`

All index sets and parameters are defined exactly as in the source data. No values are enumerated here; see the Observation for all current data.