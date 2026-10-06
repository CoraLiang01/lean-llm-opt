[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine an optimal monthly production plan for 80 products over 22 days, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quota, fixed activation costs per product, and minimum batch sizes. Production quantities must be integer multiples of 100 kg, and binary variables indicate whether each product is produced (incurring its activation cost).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is Products (i = 1 to 80, corresponding to A1–A80).
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer units of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if product i is produced (i.e., production line for product i is activated), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Maximum demand per product: from 36-1.csv, row "Maximum Demand (100 kg units)", columns A1–A80.
    -   Selling price per product: from 36-1.csv, row "Selling Price ($/100 kg)", columns A1–A80.
    -   Production cost per product: from 36-1.csv, row "Production Cost ($/100 kg)", columns A1–A80.
    -   Daily production quota per product: from 36-1.csv, row "Production Quota (max per day)", columns A1–A80.
    -   Fixed activation cost per product: from 36-2.csv, row "Activation Cost ($)", columns A1–A80.
    -   Minimum batch size per product: from 36-3.csv, row "Minimum Batch Size (100 kg units)", columns A1–A80.
    -   Number of production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as total revenue from all products minus total variable production costs and minus total fixed activation costs for all activated products:
    -   Objective = sum over i of [(Selling Price[i] - Production Cost[i]) * x[i] - Activation Cost[i] * y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, x[i] ≤ Maximum Demand[i].
    -   Constraint 2 (Production Capacity): For each product i, x[i] ≤ Daily Production Quota[i] * 22.
    -   Constraint 3 (Minimum Batch Size): For each product i, if y[i] = 1, then x[i] ≥ Minimum Batch Size[i]; if y[i] = 0, then x[i] = 0. This is enforced by: x[i] ≥ Minimum Batch Size[i] * y[i].
    -   Constraint 4 (Linking): For each product i, x[i] ≤ (Maximum Demand[i] or Daily Production Quota[i] * 22) * y[i] (to ensure x[i] = 0 if y[i] = 0).
    -   Constraint 5 (Integrality): x[i] ∈ {0, 1, 2, ...} (integer), y[i] ∈ {0, 1} (binary), for all i.
[Abstract Model Plan END]