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

2. **Assignment only to open centres:**
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\},\quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of service centres, from column "Service Center" in table_id file_1_view_0 (service_centers_fixed_costs.csv).
- $J$: set of customers, from column "Customer" in table_id file_0_view_0 (expanded_customer_service_costs.csv).
- $f_i$: fixed opening cost for centre $i$, from column "Fixed Opening Cost" in table_id file_1_view_0.
- $c_{ij}$: cost to serve customer $j$ from centre $i$, from column $i$ (centre) and row $j$ (customer) in table_id file_0_view_0.

##### Data Mapping

- $I$: All values in "Service Center" column, table_id file_1_view_0.
- $J$: All values in "Customer" column, table_id file_0_view_0.
- $f_i$: "Fixed Opening Cost" column, table_id file_1_view_0, keyed by "Service Center".
- $c_{ij}$: Entry at row with "Customer" $j$ and column $i$ (centre) in table_id file_0_view_0.
- All constraints and variables are as described above, with indices and parameters mapped directly to the CSV data.