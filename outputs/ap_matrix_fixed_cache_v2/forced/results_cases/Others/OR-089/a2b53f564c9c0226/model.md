[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which service centres to open (from ten candidates) and how to assign each of fifteen customers to exactly one open centre, so as to minimize the total cost (sum of fixed opening costs and customer–centre service costs), subject to each centre serving at most four customers and customers only being assigned to open centres.
2.  **Identify Model Type:** Based on the query, this is a capacitated facility location problem (a Mixed-Integer Programming, Fixed-Charge assignment model).
3.  **Define Index Sets:** The primary indices are Service Centres (SC = {SC1, ..., SC10}) and Customers (C = {C1, ..., C15}).
4.  **Define Decision Variables:**
    -   `y[sc]` = 1 if service centre `sc` is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[c, sc]` = 1 if customer `c` is assigned to service centre `sc`, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening costs for each centre from: 'Fixed Opening Cost' column in service_centers_fixed_costs.csv.
    -   Service cost for assigning customer to centre from: columns SC1–SC10 in expanded_customer_service_costs.csv, indexed by 'Customer'.
    -   Capacity per centre: maximum 4 customers per centre (from query, not schema).
6.  **Formulate Objective:** Minimize the sum over all centres of (fixed opening cost × y[sc]) plus the sum over all customer–centre pairs of (service cost × x[c, sc]).
7.  **Formulate Constraints:**
    -   Assignment constraint: For each customer, sum over all centres of x[c, sc] = 1 (each customer assigned to exactly one centre).
    -   Open-centre assignment constraint: For each customer and centre, x[c, sc] ≤ y[sc] (customers can only be assigned to open centres).
    -   Capacity constraint: For each centre, sum over all customers of x[c, sc] ≤ 4 (no centre serves more than 4 customers).
    -   Variable domains: y[sc] ∈ {0,1}; x[c, sc] ∈ {0,1}.
[Abstract Model Plan END]