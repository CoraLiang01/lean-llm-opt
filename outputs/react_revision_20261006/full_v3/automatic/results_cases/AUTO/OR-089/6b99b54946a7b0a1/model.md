##### Decision Variables

Let
- $y_s \in \{0,1\}$: 1 if service centre $s$ is opened, 0 otherwise, for all $s \in S$.
- $x_{sc} \in \{0,1\}$: 1 if customer $c$ is assigned to centre $s$, 0 otherwise, for all $s \in S$, $c \in C$.

##### Parameters

- $f_s$: Fixed opening cost for centre $s$ (from service_centers_fixed_costs.csv, column "Fixed Opening Cost", table_id: file_1_view_0).
- $c_{sc}$: Cost to serve customer $c$ from centre $s$ (from expanded_customer_service_costs.csv, columns SC1–SC10, rows Customer C1–C15, table_id: file_0_view_0).
- $S$: Set of service centres (from service_centers_fixed_costs.csv, column "Service Center", table_id: file_1_view_0).
- $C$: Set of customers (from expanded_customer_service_costs.csv, column "Customer", table_id: file_0_view_0).

##### Objective Function

\[
\min \sum_{s \in S} f_s y_s + \sum_{s \in S} \sum_{c \in C} c_{sc} x_{sc}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{s \in S} x_{sc} = 1 \quad \forall c \in C
   \]

2. **Assignment only to open centres:**
   \[
   x_{sc} \leq y_s \quad \forall s \in S,\, c \in C
   \]

3. **Centre capacity (at most 4 customers per open centre):**
   \[
   \sum_{c \in C} x_{sc} \leq 4 y_s \quad \forall s \in S
   \]

4. **Variable domains:**
   \[
   x_{sc} \in \{0,1\} \quad \forall s \in S,\, c \in C
   \]
   \[
   y_s \in \{0,1\} \quad \forall s \in S
   \]

---

##### Data Mapping

- $S$: All "Service Center" values in service_centers_fixed_costs.csv (table_id: file_1_view_0, column "Service Center")
- $C$: All "Customer" values in expanded_customer_service_costs.csv (table_id: file_0_view_0, column "Customer")
- $f_s$: "Fixed Opening Cost" for each $s$ in service_centers_fixed_costs.csv (table_id: file_1_view_0, columns "Service Center", "Fixed Opening Cost")
- $c_{sc}$: Entry in expanded_customer_service_costs.csv for customer $c$, centre $s$ (table_id: file_0_view_0, row "Customer" = $c$, column $s$)
- All indices and parameters are defined by the full set of rows and columns in the respective CSVs as described above.