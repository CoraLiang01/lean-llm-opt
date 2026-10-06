##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i$ is opened, 0 otherwise, for each $i$ in the set of service centres $I$.
- $x_{ij} \in \{0,1\}$: 1 if customer $j$ is assigned to service centre $i$, 0 otherwise, for each $i \in I$, $j \in J$.

##### Parameters

- $f_i$: Fixed opening cost for service centre $i$ (from service_centers_fixed_costs.csv, column "Fixed Opening Cost", table_id: file_1_view_0, row_id: $i$).
- $c_{ij}$: Cost to serve customer $j$ from service centre $i$ (from expanded_customer_service_costs.csv, table_id: file_0_view_0, row_id: $j$, column_id: $i$).
- $I$: Set of service centres (SC1–SC10, from service_centers_fixed_costs.csv, column "Service Center", table_id: file_1_view_0).
- $J$: Set of customers (C1–C15, from expanded_customer_service_costs.csv, column "Customer", table_id: file_0_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Customers assigned only to open centres:**
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $f_i$: service_centers_fixed_costs.csv, table_id: file_1_view_0, row_id: $i$, column "Fixed Opening Cost"
- $c_{ij}$: expanded_customer_service_costs.csv, table_id: file_0_view_0, row_id: $j$, column_id: $i$
- $I$: service_centers_fixed_costs.csv, table_id: file_1_view_0, column "Service Center"
- $J$: expanded_customer_service_costs.csv, table_id: file_0_view_0, column "Customer"