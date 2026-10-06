[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which suppliers to activate and how much each supermarket should order from each supplier, so that all supermarket demands are met at minimum total cost. The total cost includes both the fixed activation costs for suppliers and the per-unit transportation costs for delivering goods from suppliers to supermarkets.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Facility Location (Uncapacitated Facility Location) problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (from fixed_cost.csv and transportation_costs.csv; e.g., S1, S2)
    - Supermarkets/customers (from demand.csv and transportation_costs.csv; e.g., C1, C2)
4.  **Define Decision Variables:**
    -   `y[i]` = Binary variable indicating whether supplier i is activated (1 if supplier i is used, 0 otherwise). Type: GRB.BINARY.
    -   `x[i,j]` = Continuous variable representing the amount of goods supplied from supplier i to supermarket j. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Fixed activation costs for each supplier: from 'fixed_costs' column in fixed_cost.csv, keyed by supplier (S1, S2).
    -   Per-unit transportation costs: from transportation_costs.csv, columns 'C1', 'C2', etc., rows keyed by supplier (S1, S2).
    -   Supermarket demands: from 'demand' column in demand.csv, keyed by customer (C1, C2).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - The fixed activation costs for all suppliers that are activated (sum over i of fixed_costs[i] * y[i])
    - The total transportation costs for all goods shipped (sum over i and j of transportation_costs[i][j] * x[i,j])
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each supermarket j, the total amount received from all suppliers must equal its demand (sum over i of x[i,j] = demand[j]).
    -   Constraint 2 (Supplier Activation Linking): For each supplier i and supermarket j, the amount shipped from supplier i to supermarket j cannot be positive unless supplier i is activated (x[i,j] ≤ M * y[i], where M is a sufficiently large constant, e.g., the total demand).
    -   Constraint 3 (Nonnegativity): All shipment variables x[i,j] ≥ 0.
    -   Constraint 4 (Binary): All activation variables y[i] ∈ {0,1}.
[Abstract Model Plan END]