[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, specifically {A1, A2, ..., A80} as enumerated in the CSV schema.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced at all), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row "Selling Price ($/100 kg)", columns A1–A80.
        -   Production cost per 100 kg: from 36-1.csv, row "Production Cost ($/100 kg)", columns A1–A80.
        -   Fixed activation cost: from 36-2.csv, row "Activation Cost ($)", columns A1–A80.
    -   Constraint coefficients and RHS:
        -   Maximum demand per product: from 36-1.csv, row "Maximum Demand (100 kg units)", columns A1–A80.
        -   Daily production quota per product: from 36-1.csv, row "Production Quota (max per day)", columns A1–A80.
        -   Minimum batch size per product: from 36-3.csv, row "Minimum Batch Size (100 kg units)", columns A1–A80.
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for each product that is produced:
    -   Maximize:  
        sum over i of [(Selling Price[i] - Production Cost[i]) × x[i] - Activation Cost[i] × y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, production cannot exceed its maximum demand:
        -   x[i] ≤ Maximum Demand[i]
    -   Constraint 2 (Production Quota Limit): For each product i, production cannot exceed what is possible in 22 days at its daily quota:
        -   x[i] ≤ 22 × Production Quota[i]
    -   Constraint 3 (Minimum Batch Size): If a product is produced (y[i] = 1), at least its minimum batch size must be produced:
        -   x[i] ≥ Minimum Batch Size[i] × y[i]
    -   Constraint 4 (Activation Linking): If a product is not produced (y[i] = 0), its production quantity must be zero:
        -   x[i] ≤ Maximum Demand[i] × y[i]
    -   Constraint 5 (Shared Production Days): The sum of equivalent production days used across all products cannot exceed 22:
        -   sum over i of [x[i] / Production Quota[i]] ≤ 22
    -   Constraint 6 (Variable Types): For all i,
        -   x[i] ≥ 0 and integer
        -   y[i] ∈ {0, 1}
[Abstract Model Plan END]