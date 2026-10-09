[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from ten candidates) and how to assign each of fifteen customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to each centre serving at most four customers and customers only being assigned to open centres.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with assignment and activation (fixed-charge) decisions.
3.  **Define Index Sets:** The primary indices are:
    - Service Centres: \( S = \{\text{SC1}, \ldots, \text{SC10}\} \)
    - Customers: \( C = \{\text{C1}, \ldots, \text{C15}\} \)
4.  **Define Decision Variables:**
    -   `y[s]` = 1 if service centre \( s \) is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[c,s]` = 1 if customer \( c \) is assigned to service centre \( s \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening costs for each centre: from 'Fixed Opening Cost' column in service_centers_fixed_costs.csv, indexed by 'Service Center'.
    -   Service cost for each customer–centre pair: from the corresponding SC columns in expanded_customer_service_costs.csv, indexed by 'Customer' and service centre.
    -   Capacity per centre: fixed at 4 customers per centre (from query, not schema).
6.  **Formulate Objective:** Minimize the sum of:
    -   Total fixed opening costs: \(\sum_{s \in S} \text{Fixed Opening Cost}[s] \cdot y[s]\)
    -   Total assignment (service) costs: \(\sum_{c \in C} \sum_{s \in S} \text{Service Cost}[c,s] \cdot x[c,s]\)
7.  **Formulate Constraints:**
    -   Assignment: Each customer must be assigned to exactly one centre: \(\sum_{s \in S} x[c,s] = 1\) for all \( c \in C \).
    -   Activation linking: Customers can only be assigned to open centres: \(x[c,s] \leq y[s]\) for all \( c \in C, s \in S \).
    -   Capacity: Each centre serves at most 4 customers: \(\sum_{c \in C} x[c,s] \leq 4 \cdot y[s]\) for all \( s \in S \).
[Abstract Model Plan END]