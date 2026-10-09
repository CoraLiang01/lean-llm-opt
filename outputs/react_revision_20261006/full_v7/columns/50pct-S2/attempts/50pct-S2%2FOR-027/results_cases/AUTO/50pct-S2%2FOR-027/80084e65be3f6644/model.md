##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Customers assigned only to opened centres:**
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: set of service centres, from column "Service Center" in table_id "file_1_view_0" (service_centers_fixed_costs.csv).
- $J$: set of customers, from column "Customer" in table_id "file_0_view_0" (expanded_customer_service_costs.csv).
- $f_i$: fixed opening cost for centre $i$, from column "Fixed Opening Cost" in table_id "file_1_view_0", indexed by "Service Center".
- $c_{ij}$: cost to serve customer $j$ from centre $i$, from table_id "file_0_view_0", with row "Customer" $j$ and column $i$ ("SC1"..."SC10").
- All assignments and costs use the exact identifiers as in the source tables.

##### Data Mapping

- $I$: table_id "file_1_view_0", column "Service Center"
- $J$: table_id "file_0_view_0", column "Customer"
- $f_i$: table_id "file_1_view_0", columns "Service Center", "Fixed Opening Cost"
- $c_{ij}$: table_id "file_0_view_0", row "Customer" $j$, column $i$ ("SC1"..."SC10")
- $x_{ij}$, $y_i$: as defined above, indexed over $I$ and $J$ per the above mappings

All parameters and sets are defined directly from the CSV data as described.