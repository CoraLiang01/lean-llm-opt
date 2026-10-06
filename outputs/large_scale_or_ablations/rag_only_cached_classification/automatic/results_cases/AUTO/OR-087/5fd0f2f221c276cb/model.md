[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine an optimal monthly production plan for 80 products over 22 days, maximizing total profit. The plan must account for product-specific maximum demand, selling prices, production costs, daily production quotas, fixed activation costs for each production line, and minimum batch sizes. All production and activation decisions must be integer (multiples of 100 kg and binary, respectively).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is Products (i = 1,...,80), corresponding to A1–A80.
4.  **Define Decision Variables:**
    -   `x[i]` = Total quantity of product i to produce in the month (in integer units of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced at all), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   From 36-1.csv:
        -   Maximum Demand: Row "Maximum Demand (100 kg units)", columns A1–A80.
        -   Selling Price: Row "Selling Price ($/100 kg)", columns A1–A80.
        -   Production Cost: Row "Production Cost ($/100 kg)", columns A1–A80.
        -   Production Quota (max per day): Row "Production Quota (max per day)", columns A1–A80.
    -   From 36-2.csv:
        -   Activation Cost: Row "Activation Cost ($)", columns A1–A80.
    -   From 36-3.csv:
        -   Minimum Batch Size: Row "Minimum Batch Size (100 kg units)", columns A1–A80.
    -   Number of production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as:
        sum over i of [ (Selling Price[i] - Production Cost[i]) * x[i] - Activation Cost[i] * y[i] ]
    That is, total revenue minus total variable production costs minus total fixed activation costs.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, x[i] ≤ Maximum Demand[i].
    -   Constraint 2 (Production Capacity): For each product i, x[i] ≤ Production Quota[i] * 22.
    -   Constraint 3 (Minimum Batch Size): For each product i, if y[i] = 1, then x[i] ≥ Minimum Batch Size[i]; if y[i] = 0, then x[i] = 0. This is enforced by: x[i] ≥ Minimum Batch Size[i] * y[i].
    -   Constraint 4 (Linking): For each product i, x[i] ≤ (Production Quota[i] * 22) * y[i]. (Ensures x[i]=0 if y[i]=0.)
    -   Constraint 5 (Integrality): All x[i] are integer multiples of 100 kg (enforced by integer variable), and all y[i] are binary.
[Abstract Model Plan END]