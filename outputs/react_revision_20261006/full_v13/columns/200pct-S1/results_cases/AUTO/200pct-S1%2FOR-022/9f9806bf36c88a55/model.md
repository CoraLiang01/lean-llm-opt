##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   x_{ij} \leq D_j y_i, \quad \forall i \in I,\, j \in J
   \]
   where $D_j$ is the demand of branch $j$ (from data), ensuring $x_{ij}=0$ if $y_i=0$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility IDs from file_1_view_0 and file_2_view_0: S1, S2, S3, S4, S5)
- $J$: Set of branches/customers (customer IDs from file_0_view_0 and file_2_view_0: C1, C2, C3, C4, C5)
- $d_j$: Demand of branch $j$ (column "demand_units" in file_0_view_0, indexed by "customer_id")
- $f_i$: Fixed opening cost for supplier $i$ (column "fixed_opening_cost" in file_1_view_0, indexed by "facility_id")
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$ (column "transportation_cost_to_{j}" in file_2_view_0, indexed by "facility_id")
- $D_j$: Demand of branch $j$ (same as $d_j$)

##### Data Mapping

- $I$: All "facility_id" in file_1_view_0 and file_2_view_0
- $J$: All "customer_id" in file_0_view_0 and columns with suffix "transportation_cost_to_{j}" in file_2_view_0
- $d_j$: file_0_view_0, columns "customer_id", "demand_units"
- $f_i$: file_1_view_0, columns "facility_id", "fixed_opening_cost"
- $c_{ij}$: file_2_view_0, row "facility_id", columns "transportation_cost_to_{j}" (mapping to $j$ in $J$)

All index sets and parameters are defined directly from the current CSV data. No capacity limits are imposed except those implied by demand and activation logic.