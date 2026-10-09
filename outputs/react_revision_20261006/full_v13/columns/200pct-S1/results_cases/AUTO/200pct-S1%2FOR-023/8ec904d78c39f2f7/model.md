##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a valid upper bound on total shipments from any supplier (since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from fixed_cost.csv, column "Unnamed: 3", table_id: file_1_view_0)
- $J$: Set of stores (from demand.csv, column "Customer", table_id: file_0_view_0)
- $d_j$: Demand of store $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$: Fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, row "Unnamed: 4" for supplier $i$, column header for store $j$, table_id: file_2_view_0)
- $M = \sum_{j \in J} d_j$ (sum over all store demands)

##### Data Mapping

- $I$: All unique values in file_1_view_0, column "Unnamed: 3"
- $J$: All unique values in file_0_view_0, column "Customer"
- $d_j$: file_0_view_0, column "demand", keyed by "Customer"
- $f_i$: file_1_view_0, column "fixed_costs", keyed by "Unnamed: 3"
- $c_{ij}$: file_2_view_0, row "Unnamed: 4" (supplier $i$), column header (store $j$)
- $M$: $\sum_{j \in J} d_j$ (sum of all values in file_0_view_0, column "demand")

No additional constraints or bounds are imposed beyond those above. All index sets and parameters are defined directly from the current CSV data.