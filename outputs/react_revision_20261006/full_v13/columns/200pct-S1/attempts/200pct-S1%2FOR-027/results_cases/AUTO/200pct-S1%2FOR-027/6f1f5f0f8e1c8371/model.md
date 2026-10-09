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

2. **Customers assigned only to open centres:**
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   y_i \in \{0,1\},\quad x_{ij} \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of service centres, from column "Service Center" in `service_centers_fixed_costs.csv` (`file_1_view_0`).
- $J$: set of customers, from column "Customer" in `expanded_customer_service_costs.csv` (`file_0_view_0`).
- $f_i$: fixed opening cost for centre $i$, from column "Fixed Opening Cost" in `service_centers_fixed_costs.csv` (`file_1_view_0`).
- $c_{ij}$: cost to serve customer $j$ from centre $i$, from entry at row "Customer" $j$, column $i$ in `expanded_customer_service_costs.csv` (`file_0_view_0`).

##### Data Mapping

- $I$: All values in column "Service Center" of table_id `file_1_view_0`.
- $J$: All values in column "Customer" of table_id `file_0_view_0`.
- $f_i$: "Fixed Opening Cost" for each "Service Center" in table_id `file_1_view_0`.
- $c_{ij}$: Entry at row "Customer" $j$, column $i$ in table_id `file_0_view_0`.
- All constraints and variables are defined for these sets and parameters as above.