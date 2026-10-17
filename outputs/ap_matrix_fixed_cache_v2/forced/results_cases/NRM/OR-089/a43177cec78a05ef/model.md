#### Index Sets

- $S$: set of service centres (from service_centers_fixed_costs.csv, column "Service Center")
- $C$: set of customers (from expanded_customer_service_costs.csv, column "Customer")

#### Parameters

- $f_s$: fixed opening cost for centre $s \in S$ (from service_centers_fixed_costs.csv, column "Fixed Opening Cost")
- $c_{cs}$: cost to serve customer $c \in C$ from centre $s \in S$ (from expanded_customer_service_costs.csv, columns "SC1"–"SC10" for each row "Customer")

#### Decision Variables

- $y_s \in \{0,1\}$: 1 if centre $s$ is opened, 0 otherwise
- $x_{cs} \in \{0,1\}$: 1 if customer $c$ is assigned to centre $s$, 0 otherwise

#### Objective

$$
\min \quad \sum_{s \in S} f_s y_s + \sum_{c \in C} \sum_{s \in S} c_{cs} x_{cs}
$$

#### Constraints

1. **Each customer assigned to exactly one centre:**
   $$
   \sum_{s \in S} x_{cs} = 1 \quad \forall c \in C
   $$

2. **Customers assigned only to opened centres:**
   $$
   x_{cs} \leq y_s \quad \forall c \in C, \forall s \in S
   $$

3. **Each opened centre serves at most 4 customers:**
   $$
   \sum_{c \in C} x_{cs} \leq 4 y_s \quad \forall s \in S
   $$

4. **Variable domains:**
   $$
   y_s \in \{0,1\} \quad \forall s \in S
   $$
   $$
   x_{cs} \in \{0,1\} \quad \forall c \in C, \forall s \in S
   $$

---

#### Data Mapping

- $S$ (service centres), $f_s$ (fixed opening cost): from service_centers_fixed_costs.csv, columns "Service Center", "Fixed Opening Cost", table_id file_1_view_0.
- $C$ (customers), $c_{cs}$ (service cost): from expanded_customer_service_costs.csv, columns "Customer", "SC1"–"SC10", table_id file_0_view_0.