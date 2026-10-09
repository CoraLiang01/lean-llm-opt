##### Decision Variables

- $y_s \in \{0,1\}$: 1 if service centre $s$ is opened, 0 otherwise, for each $s \in S$.
- $x_{sc} \in \{0,1\}$: 1 if customer $c$ is assigned to service centre $s$, 0 otherwise, for each $s \in S$, $c \in C$.

##### Objective Function

\[
\min \sum_{s \in S} f_s y_s + \sum_{s \in S} \sum_{c \in C} c_{sc} x_{sc}
\]

where:
- $f_s$ is the fixed opening cost for service centre $s$,
- $c_{sc}$ is the cost to serve customer $c$ from service centre $s$.

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{s \in S} x_{sc} = 1 \quad \forall c \in C
   \]

2. **Customers assigned only to open centres:**
   \[
   x_{sc} \leq y_s \quad \forall s \in S,\, c \in C
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{c \in C} x_{sc} \leq 4 y_s \quad \forall s \in S
   \]

4. **Variable domains:**
   \[
   y_s \in \{0,1\} \quad \forall s \in S
   \]
   \[
   x_{sc} \in \{0,1\} \quad \forall s \in S,\, c \in C
   \]

##### Index Sets and Data Mapping

- $S$: set of service centres, from column "Service Center" in table_id="file_1_view_0" (/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv)
- $C$: set of customers, from column "Customer" in table_id="file_0_view_0" (/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv)
- $f_s$: fixed opening cost for centre $s$, from column "Fixed Opening Cost" in table_id="file_1_view_0"
- $c_{sc}$: service cost for assigning customer $c$ to centre $s$, from table_id="file_0_view_0", row "Customer" = $c$, column $s$ (SC1–SC10)

All parameters are mapped directly to the supplied CSV data as described above. No values are enumerated here.