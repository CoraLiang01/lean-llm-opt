[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from ten candidates) and how to assign each of fifteen customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to each centre serving at most four customers and customers only being assigned to open centres.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) facility location problem with assignment and capacity constraints (specifically, a capacitated uncapacitated facility location problem with single assignment).
3.  **Define Index Sets:** The primary indices are:
    - Service Centres: \( S = \{\text{SC1}, \ldots, \text{SC10}\} \)
    - Customers: \( C = \{\text{C1}, \ldots, \text{C15}\} \)
4.  **Define Decision Variables:**
    -   \( y_s \) = 1 if service centre \( s \) is opened, 0 otherwise. Type: GRB.BINARY.
    -   \( x_{c,s} \) = 1 if customer \( c \) is assigned to centre \( s \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening costs for each centre from column 'Fixed Opening Cost' in service_centers_fixed_costs.csv, indexed by 'Service Center'.
    -   Service cost for assigning customer \( c \) to centre \( s \) from expanded_customer_service_costs.csv, columns SC1–SC10, indexed by 'Customer'.
    -   Capacity per centre: maximum 4 customers per centre (given in query, not in CSV).
6.  **Formulate Objective:** Minimize the sum of fixed opening costs for all opened centres plus the sum of service costs for all customer–centre assignments, i.e., minimize \(\sum_{s \in S} \text{FixedCost}_s \cdot y_s + \sum_{c \in C} \sum_{s \in S} \text{ServiceCost}_{c,s} \cdot x_{c,s}\).
7.  **Formulate Constraints:**
    -   Assignment constraint: Each customer must be assigned to exactly one centre: for all \( c \in C \), \(\sum_{s \in S} x_{c,s} = 1\).
    -   Open-centre assignment constraint: Customers can only be assigned to open centres: for all \( c \in C, s \in S \), \(x_{c,s} \leq y_s\).
    -   Capacity constraint: Each centre can serve at most 4 customers: for all \( s \in S \), \(\sum_{c \in C} x_{c,s} \leq 4\).
    -   Variable domains: \(y_s \in \{0,1\}\), \(x_{c,s} \in \{0,1\}\).
[Abstract Model Plan END]