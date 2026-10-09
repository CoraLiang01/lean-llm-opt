##### Decision Variables

$y_s \in \{0,1\}$: 1 if service centre $s$ is opened, 0 otherwise, for all $s \in S$.

$x_{sc} \in \{0,1\}$: 1 if customer $c$ is assigned to centre $s$, 0 otherwise, for all $s \in S$, $c \in C$.

##### Objective Function

$\min \sum_{s \in S} f_s y_s + \sum_{s \in S} \sum_{c \in C} c_{sc} x_{sc}$

##### Constraints

1. Each customer is assigned to exactly one centre:
   $$
   \sum_{s \in S} x_{sc} = 1, \quad \forall c \in C
   $$

2. Customers can only be assigned to open centres:
   $$
   x_{sc} \leq y_s, \quad \forall s \in S, \forall c \in C
   $$

3. Each centre serves at most 4 customers:
   $$
   \sum_{c \in C} x_{sc} \leq 4 y_s, \quad \forall s \in S
   $$

4. Variable domains:
   $$
   y_s \in \{0,1\}, \quad \forall s \in S
   $$
   $$
   x_{sc} \in \{0,1\}, \quad \forall s \in S, \forall c \in C
   $$

##### Index Sets and Data Mapping

- $S$: set of service centres, from column "Service Center" in table_id file_1_view_0 (service_centers_fixed_costs.csv).
- $C$: set of customers, from column "Customer" in table_id file_0_view_0 (expanded_customer_service_costs.csv).
- $f_s$: fixed opening cost for centre $s$, from column "Fixed Opening Cost" in table_id file_1_view_0, keyed by "Service Center".
- $c_{sc}$: cost to serve customer $c$ from centre $s$, from table_id file_0_view_0, row "Customer" $c$, column $s$.
- $x_{sc}$: assignment variable for customer $c$ to centre $s$.
- $y_s$: open/close variable for centre $s$.

All parameters and sets are mapped directly from the CSV files as described above.