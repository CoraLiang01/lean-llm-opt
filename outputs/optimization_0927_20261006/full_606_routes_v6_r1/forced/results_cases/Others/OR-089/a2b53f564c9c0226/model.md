[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from ten candidates) and how to assign each of fifteen customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to each centre serving at most four customers and customers only being assigned to open centres.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) facility location problem with assignment and capacity constraints (specifically, a capacitated uncapacitated facility location problem with single assignment).
3.  **Define Index Sets:** The primary indices are:
    - Service Centres: \( S = \{\text{SC1}, \ldots, \text{SC10}\} \) (from 'Service Center' in service_centers_fixed_costs.csv)
    - Customers: \( C = \{\text{C1}, \ldots, \text{C15}\} \) (from 'Customer' in expanded_customer_service_costs.csv)
4.  **Define Decision Variables:**
    -   \( y_s \) = 1 if service centre \( s \) is opened, 0 otherwise. Type: GRB.BINARY.
    -   \( x_{cs} \) = 1 if customer \( c \) is assigned to service centre \( s \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each centre: from 'Fixed Opening Cost' column in service_centers_fixed_costs.csv, indexed by 'Service Center'.
    -   Service cost for each customer–centre pair: from the SC columns in expanded_customer_service_costs.csv, indexed by 'Customer' and service centre.
    -   Capacity per centre: fixed at 4 customers per centre (from query, not schema).
6.  **Formulate Objective:** Minimize the sum of fixed opening costs for all opened centres plus the sum of service costs for all customer–centre assignments, i.e., minimize \(\sum_{s \in S} \text{FixedCost}_s \cdot y_s + \sum_{c \in C} \sum_{s \in S} \text{ServiceCost}_{cs} \cdot x_{cs}\).
7.  **Formulate Constraints:**
    -   Assignment constraint: Each customer must be assigned to exactly one centre: for all \( c \in C \), \(\sum_{s \in S} x_{cs} = 1\).
    -   Open-centre assignment constraint: A customer can only be assigned to an open centre: for all \( c \in C, s \in S \), \( x_{cs} \leq y_s \).
    -   Capacity constraint: Each centre can serve at most 4 customers: for all \( s \in S \), \(\sum_{c \in C} x_{cs} \leq 4\).
    -   Variable domains: \( y_s \in \{0,1\} \), \( x_{cs} \in \{0,1\} \).
[Abstract Model Plan END]