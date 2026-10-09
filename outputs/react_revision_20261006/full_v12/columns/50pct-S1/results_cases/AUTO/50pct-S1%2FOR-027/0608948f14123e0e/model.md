##### Decision Variables

$y_i \in \{0,1\}$: 1 if service centre $i$ is opened, 0 otherwise, for each $i \in I$ (service centres from service_centers_fixed_costs.csv, column "Service Center").
$x_{ij} \in \{0,1\}$: 1 if customer $j$ is assigned to centre $i$, 0 otherwise, for each $i \in I$, $j \in J$ (customers from expanded_customer_service_costs.csv, column "Customer").

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

where:
- $f_i$ is the fixed opening cost of centre $i$ (service_centers_fixed_costs.csv, column "Fixed Opening Cost", table_id: file_1_view_0).
- $c_{ij}$ is the cost to serve customer $j$ from centre $i$ (expanded_customer_service_costs.csv, columns SC1–SC10, table_id: file_0_view_0).

##### Constraints

1. Each customer is assigned to exactly one centre:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$
2. Customers can only be assigned to open centres:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$
3. Each centre serves at most 4 customers:
   $$
   \sum_{j \in J} x_{ij} \leq 4 y_i, \quad \forall i \in I
   $$
4. Variable domains:
   $$
   y_i \in \{0,1\},\quad x_{ij} \in \{0,1\}
   $$

##### Index Sets

- $I$: Service centres from service_centers_fixed_costs.csv, column "Service Center", table_id: file_1_view_0.
- $J$: Customers from expanded_customer_service_costs.csv, column "Customer", table_id: file_0_view_0.

##### Data Mapping

- $f_i$: service_centers_fixed_costs.csv, table_id: file_1_view_0, columns "Service Center", "Fixed Opening Cost".
- $c_{ij}$: expanded_customer_service_costs.csv, table_id: file_0_view_0, row "Customer" $j$, column $i$ (SC1–SC10).
- $x_{ij}$: assignment of customer $j$ to centre $i$.
- $y_i$: open/close status of centre $i$.

All parameters and sets are defined directly from the current CSV data as described above.