[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 specific products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, specifically {A1, A2, ..., A80} as enumerated in the query and CSV schema.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer multiples of 100 kg units). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced at all), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per product: from 36-1.csv, row "Selling Price ($/100 kg)".
        -   Production cost per product: from 36-1.csv, row "Production Cost ($/100 kg)".
        -   Fixed activation cost per product: from 36-2.csv, row "Activation Cost ($)".
    -   Constraint coefficients:
        -   Maximum demand per product: from 36-1.csv, row "Maximum Demand (100 kg units)".
        -   Daily production quota per product: from 36-1.csv, row "Production Quota (max per day)".
        -   Minimum batch size per product: from 36-3.csv, row "Minimum Batch Size (100 kg units)".
    -   Shared resource:
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for all activated products:
    -   Maximize:  
        sum over i of [(Selling Price[i] - Production Cost[i]) × x[i] - Activation Cost[i] × y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, x[i] ≤ Maximum Demand[i].
    -   Constraint 2 (Production Quota Limit): For each product i, x[i] ≤ 22 × Production Quota[i] (cannot exceed what could be produced in 22 days if the line is dedicated to product i).
    -   Constraint 3 (Minimum Batch Size): For each product i, if product i is produced (y[i]=1), x[i] ≥ Minimum Batch Size[i]; if not produced (y[i]=0), x[i]=0. This is enforced by: x[i] ≥ Minimum Batch Size[i] × y[i].
    -   Constraint 4 (Activation Linking): For each product i, x[i] ≤ Maximum Demand[i] × y[i] (or a similar large upper bound), ensuring x[i]=0 if y[i]=0.
    -   Constraint 5 (Shared Production Days): The sum over all products of (x[i] / Production Quota[i]) ≤ 22. This ensures that the total equivalent production days used across all products does not exceed the monthly limit.
    -   Constraint 6 (Integrality): All x[i] are integer (≥0), all y[i] are binary (0 or 1).
[Abstract Model Plan END]