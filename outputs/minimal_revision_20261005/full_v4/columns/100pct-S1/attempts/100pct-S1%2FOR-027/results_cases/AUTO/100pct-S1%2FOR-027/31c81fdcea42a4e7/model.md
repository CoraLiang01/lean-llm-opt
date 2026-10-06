##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Parameters

- $f_i$: Fixed opening cost for centre $i$ (from service_centers_fixed_costs.csv, column "Fixed Opening Cost", table_id: file_1_view_0).
- $c_{ij}$: Cost to serve customer $j$ from centre $i$ (from expanded_customer_service_costs.csv, columns SC1–SC10, table_id: file_0_view_0).
- $I$: Set of service centres (SC1–SC10, from service_centers_fixed_costs.csv, column "Service Center", table_id: file_1_view_0).
- $J$: Set of customers (C1–C15, from expanded_customer_service_costs.csv, column "Customer", table_id: file_0_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Customers assigned only to opened centres:**
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
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

---

#### Data Mapping

- $I$ (service centres): service_centers_fixed_costs.csv, column "Service Center", table_id: file_1_view_0
- $J$ (customers): expanded_customer_service_costs.csv, column "Customer", table_id: file_0_view_0
- $f_i$: service_centers_fixed_costs.csv, column "Fixed Opening Cost", table_id: file_1_view_0
- $c_{ij}$: expanded_customer_service_costs.csv, columns SC1–SC10 (centre $i$), rows indexed by "Customer" (customer $j$), table_id: file_0_view_0