[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which warehouses to activate and how much inventory each musician or band should source from each warehouse, in order to meet all demands at minimum total cost (including both fixed warehouse activation costs and per-unit transportation costs).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (or uncapacitated if no warehouse capacity is given) / fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (Facilities): F = {S1, S2, S3} (from fixed_cost.csv and transportation_costs.csv, 'Unnamed: 0' column)
    - Customers (Musicians/Bands): C = {C1, C2, C3} (from demand.csv and transportation_costs.csv columns)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Quantity of goods supplied from warehouse i ∈ F to customer j ∈ C. Type: GRB.CONTINUOUS (≥ 0).
    -   `y[i]` = 1 if warehouse i ∈ F is activated (operational), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed cost for activating warehouse i: from fixed_cost.csv, column 'fixed_costs', indexed by 'Unnamed: 0' (S1, S2, S3).
    -   Per-unit transportation cost from warehouse i to customer j: from transportation_costs.csv, columns 'C1', 'C2', 'C3', indexed by 'Unnamed: 0' (S1, S2, S3).
    -   Demand for each customer j: from demand.csv, column 'demand', indexed by 'customer' (C1, C2, C3).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - All fixed costs for activated warehouses: sum over i of (fixed_cost[i] * y[i])
    - All transportation costs: sum over i, j of (transportation_cost[i][j] * x[i,j])
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer j ∈ C, the total goods received from all warehouses must meet their demand: sum over i of x[i,j] = demand[j].
    -   Constraint 2 (Warehouse Activation Linking): For each warehouse i ∈ F and customer j ∈ C, x[i,j] ≤ demand[j] * y[i]. (This ensures that if any goods are shipped from warehouse i, it must be activated; demand[j] is a valid upper bound since no customer can receive more than their total demand from any warehouse.)
    -   Constraint 3 (Non-negativity): x[i,j] ≥ 0 for all i, j.
    -   Constraint 4 (Binary): y[i] ∈ {0,1} for all i.
[Abstract Model Plan END]