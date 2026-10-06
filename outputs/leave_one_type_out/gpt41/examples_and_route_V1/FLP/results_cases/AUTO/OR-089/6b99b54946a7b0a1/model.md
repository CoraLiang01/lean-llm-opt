[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from 10 candidates) and how to assign each of 15 customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to: (a) each customer is assigned to one open centre, (b) a centre can serve at most 4 customers, and (c) customers can only be assigned to centres that are open.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with assignment and fixed-charge (opening) costs.
3.  **Define Index Sets:** The primary indices are:
    - Service Centres: SC = {SC1, SC2, ..., SC10} (from 'Service Center' in service_centers_fixed_costs.csv)
    - Customers: C = {C1, C2, ..., C15} (from 'Customer' in expanded_customer_service_costs.csv)
4.  **Define Decision Variables:**
    -   `y[j]` = 1 if service centre j ∈ SC is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = 1 if customer i ∈ C is assigned to centre j ∈ SC, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each centre: from 'Fixed Opening Cost' column in service_centers_fixed_costs.csv, indexed by 'Service Center'.
    -   Service cost for each customer–centre pair: from the SC1–SC10 columns in expanded_customer_service_costs.csv, indexed by 'Customer' and 'SC'.
    -   Capacity per centre: fixed at 4 customers per centre (from query, not CSV).
6.  **Formulate Objective:** Minimize total cost = sum of fixed opening costs for all opened centres (sum over j of Fixed Opening Cost[j] * y[j]) plus sum of service costs for all customer–centre assignments (sum over i,j of ServiceCost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Assignment): For every customer i ∈ C, sum over j ∈ SC of x[i,j] = 1. (Each customer assigned to exactly one centre.)
    -   Constraint 2 (Open-to-Assign Linking): For every i ∈ C and j ∈ SC, x[i,j] ≤ y[j]. (Customers can only be assigned to open centres.)
    -   Constraint 3 (Centre Capacity): For every centre j ∈ SC, sum over i ∈ C of x[i,j] ≤ 4. (No centre serves more than 4 customers.)
    -   Constraint 4 (Variable Domains): y[j] ∈ {0,1} for all j ∈ SC; x[i,j] ∈ {0,1} for all i ∈ C, j ∈ SC.
[Abstract Model Plan END]