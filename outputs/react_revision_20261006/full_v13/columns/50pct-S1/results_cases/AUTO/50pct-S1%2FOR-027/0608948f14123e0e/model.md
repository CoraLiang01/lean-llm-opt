##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Parameters

- $f_i$: Fixed opening cost for centre $i$ (from column "Fixed Opening Cost" in table_id file_1_view_0, key "Service Center").
- $c_{ij}$: Cost to serve customer $j$ from centre $i$ (from table_id file_0_view_0, row "Customer" $j$, column $i$).
- $I$: Set of service centres (all "Service Center" in file_1_view_0 and all SC columns in file_0_view_0).
- $J$: Set of customers (all "Customer" in file_0_view_0).
- Each centre may serve at most 4 customers.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Assignment only to open centres:**
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

3. **Centre capacity (at most 4 customers per open centre):**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All "Service Center" values in file_1_view_0 and all SC columns in file_0_view_0.
- $J$: All "Customer" values in file_0_view_0.
- $f_i$: "Fixed Opening Cost" column in file_1_view_0, indexed by "Service Center".
- $c_{ij}$: file_0_view_0, row "Customer" $j$, column $i$ (SC1–SC10).
- All constraints and variables are defined over these sets and parameters as mapped.