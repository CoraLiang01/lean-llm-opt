[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is Products, denoted as i ∈ {A1, A2, ..., A80}.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)'.
        -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)'.
        -   Fixed activation cost per product: from 36-2.csv, row 'Activation Cost ($)'.
    -   Constraint coefficients:
        -   Maximum demand per product: from 36-1.csv, row 'Maximum Demand (100 kg units)'.
        -   Daily production quota per product: from 36-1.csv, row 'Production Quota (max per day)'.
        -   Minimum batch size per product: from 36-3.csv, row 'Minimum Batch Size (100 kg units)'.
    -   Shared resource:
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × production quantity] minus the fixed activation cost for each product that is produced:
    -   Maximize:  
        sum over i of [ (SellingPrice[i] - ProductionCost[i]) * x[i] - ActivationCost[i] * y[i] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, production cannot exceed its maximum demand:
        -   x[i] ≤ MaximumDemand[i]
    -   Constraint 2 (Production Quota Limit): For each product i, production cannot exceed the maximum possible in 22 days at its daily quota:
        -   x[i] ≤ 22 × ProductionQuota[i]
    -   Constraint 3 (Minimum Batch Size): If a product is produced (y[i]=1), its production must be at least the minimum batch size:
        -   x[i] ≥ MinimumBatchSize[i] × y[i]
    -   Constraint 4 (Activation Linking): If a product is not produced (y[i]=0), its production must be zero; if produced, cannot exceed demand:
        -   x[i] ≤ MaximumDemand[i] × y[i]
    -   Constraint 5 (Shared Production Days): The sum of equivalent production days used by all products cannot exceed 22. For each product, producing one unit uses 1/(ProductionQuota[i]) days:
        -   sum over i of ( x[i] / ProductionQuota[i] ) ≤ 22
    -   Constraint 6 (Integrality): All x[i] are integer ≥ 0; all y[i] are binary (0 or 1).
[Abstract Model Plan END]