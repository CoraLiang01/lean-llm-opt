Mathematical Model

Sets:
- Let \( S = \{\text{SC1}, \text{SC2}, \ldots, \text{SC10}\} \) be the set of service centres, as listed in service_centers_fixed_costs.csv (table_id: file_1_view_0, column: Service Center).
- Let \( C = \{\text{C1}, \text{C2}, \ldots, \text{C15}\} \) be the set of customers, as listed in expanded_customer_service_costs.csv (table_id: file_0_view_0, column: Customer).

Parameters:
- \( f_s \): Fixed opening cost for centre \( s \in S \), from service_centers_fixed_costs.csv (table_id: file_1_view_0, columns: Service Center, Fixed Opening Cost).
- \( c_{cs} \): Cost to serve customer \( c \in C \) from centre \( s \in S \), from expanded_customer_service_costs.csv (table_id: file_0_view_0, columns: Customer, SC1–SC10).

Decision Variables:
- \( y_s \in \{0,1\} \): 1 if centre \( s \) is opened, 0 otherwise.
- \( x_{cs} \in \{0,1\} \): 1 if customer \( c \) is assigned to centre \( s \), 0 otherwise.

Objective:
\[
\min \sum_{s \in S} f_s y_s + \sum_{c \in C} \sum_{s \in S} c_{cs} x_{cs}
\]

Subject to:
1. Each customer is assigned to exactly one centre:
\[
\forall c \in C: \quad \sum_{s \in S} x_{cs} = 1
\]

2. Customers can only be assigned to opened centres:
\[
\forall c \in C, \forall s \in S: \quad x_{cs} \leq y_s
\]

3. Each centre serves at most 4 customers:
\[
\forall s \in S: \quad \sum_{c \in C} x_{cs} \leq 4
\]

4. Variable domains:
\[
y_s \in \{0,1\} \quad \forall s \in S
\]
\[
x_{cs} \in \{0,1\} \quad \forall c \in C, s \in S
\]

Data Mapping

- \( S \): All values in service_centers_fixed_costs.csv (table_id: file_1_view_0, column: Service Center)
- \( C \): All values in expanded_customer_service_costs.csv (table_id: file_0_view_0, column: Customer)
- \( f_s \): service_centers_fixed_costs.csv (table_id: file_1_view_0, columns: Service Center, Fixed Opening Cost)
- \( c_{cs} \): expanded_customer_service_costs.csv (table_id: file_0_view_0, columns: Customer, SC1–SC10)