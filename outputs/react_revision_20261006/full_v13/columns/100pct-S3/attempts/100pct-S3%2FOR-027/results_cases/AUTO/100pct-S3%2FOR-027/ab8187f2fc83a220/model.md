##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Parameters

- $f_i$: Fixed opening cost for centre $i$ (from column "Fixed Opening Cost" in table_id file_1_view_0, key "Service Center").
- $c_{ij}$: Cost to serve customer $j$ from centre $i$ (from table_id file_0_view_0, row "Customer" $j$, column $i$).
- $I$: Set of service centres (from "Service Center" in file_1_view_0 and columns SC1–SC10 in file_0_view_0).
- $J$: Set of customers (from "Customer" in file_0_view_0).

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
   \sum_{j \in J} x_{ij} \leq 4 y_i, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All service centres from column "Service Center" in table_id file_1_view_0 and columns SC1–SC10 in file_0_view_0.
- $J$: All customers from column "Customer" in table_id file_0_view_0.
- $f_i$: "Fixed Opening Cost" for each $i$ from table_id file_1_view_0, key "Service Center".
- $c_{ij}$: Entry in table_id file_0_view_0, row "Customer" $j$, column $i$.
- $x_{ij}$, $y_i$: Decision variables as defined above.