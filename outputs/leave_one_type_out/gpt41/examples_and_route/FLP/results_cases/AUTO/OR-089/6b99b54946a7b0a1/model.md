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
    -   Service cost for each customer–centre pair: from the SC1–SC10 columns in expanded_customer_service_costs.csv, indexed by 'Customer' and centre.
    -   Capacity per centre: 4 customers (given in the query, not in the CSV).
6.  **Formulate Objective:** Minimize total cost = sum over all centres of (fixed opening cost * y[j]) + sum over all customer–centre pairs of (service cost * x[i,j]).
7.  **Formulate Constraints:**
    -   Assignment constraint: For each customer i, sum over all centres j of x[i,j] = 1 (each customer assigned to exactly one centre).
    -   Open-to-assign constraint: For each customer i and centre j, x[i,j] ≤ y[j] (customers can only be assigned to open centres).
    -   Capacity constraint: For each centre j, sum over all customers i of x[i,j] ≤ 4 (no centre serves more than 4 customers).
    -   Variable domains: y[j] ∈ {0,1} for all j; x[i,j] ∈ {0,1} for all i, j.
[Abstract Model Plan END]