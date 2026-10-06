[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from 10 candidates) and how to assign each of 15 customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to: (a) each customer is assigned to one open centre, (b) a centre can serve at most 4 customers, and (c) customers can only be assigned to centres that are open.
2.  **Identify Model Type:** Based on the query, this is a capacitated facility location problem (a type of Mixed-Integer Programming, MIP, with fixed-charge and assignment structure).
3.  **Define Index Sets:** The primary indices are:
    - Service Centres: SC = {SC1, SC2, ..., SC10} (from 'Service Center' in service_centers_fixed_costs.csv)
    - Customers: C = {C1, C2, ..., C15} (from 'Customer' in expanded_customer_service_costs.csv)
4.  **Define Decision Variables:**
    -   `y[j]` = 1 if service centre j is opened, 0 otherwise. Type: GRB.BINARY, for each j in SC.
    -   `x[i,j]` = 1 if customer i is assigned to centre j, 0 otherwise. Type: GRB.BINARY, for each i in C, j in SC.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each centre: from 'Fixed Opening Cost' column in service_centers_fixed_costs.csv, indexed by 'Service Center'.
    -   Service cost for each customer–centre pair: from the columns SC1–SC10 in expanded_customer_service_costs.csv, indexed by 'Customer' and centre.
    -   Capacity per centre: 4 customers per centre (given in the query, not in the CSV).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - Fixed opening costs for all opened centres: sum over j of (Fixed Opening Cost[j] * y[j])
    - Service costs for all customer–centre assignments: sum over i, j of (Service Cost[i,j] * x[i,j])
    - So, Objective: Minimize sum_j (Fixed Opening Cost[j] * y[j]) + sum_i sum_j (Service Cost[i,j] * x[i,j])
7.  **Formulate Constraints:**
    -   Constraint 1 (Assignment): Each customer must be assigned to exactly one centre:
        - For all i in C: sum over j in SC of x[i,j] = 1
    -   Constraint 2 (Open centre only): Customers can only be assigned to open centres:
        - For all i in C, j in SC: x[i,j] ≤ y[j]
    -   Constraint 3 (Centre capacity): Each centre can serve at most 4 customers:
        - For all j in SC: sum over i in C of x[i,j] ≤ 4
    -   Constraint 4 (Variable domains): y[j] ∈ {0,1}; x[i,j] ∈ {0,1}
[Abstract Model Plan END]