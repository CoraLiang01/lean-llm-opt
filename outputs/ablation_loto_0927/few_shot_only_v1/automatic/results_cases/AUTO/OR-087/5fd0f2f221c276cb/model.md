[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal production plan for 80 products over a 22-day month, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production quantities must be integer multiples of 100 kg, and each product's production line can be either activated or not (binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is Products, denoted as i ∈ {A1, A2, ..., A80}.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce (in units of 100 kg). Type: GRB.INTEGER (non-negative).
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row "Selling Price ($/100 kg)", columns A1–A80.
        -   Production cost per 100 kg: from 36-1.csv, row "Production Cost ($/100 kg)", columns A1–A80.
        -   Fixed activation cost: from 36-2.csv, row "Activation Cost ($)", columns A1–A80.
    -   Constraint coefficients:
        -   Maximum demand: from 36-1.csv, row "Maximum Demand (100 kg units)", columns A1–A80.
        -   Daily production quota: from 36-1.csv, row "Production Quota (max per day)", columns A1–A80.
        -   Minimum batch size: from 36-3.csv, row "Minimum Batch Size (100 kg units)", columns A1–A80.
    -   Shared resource:
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × production quantity] minus the fixed activation cost for each activated product:
    -   Maximize:  
        sum over i of [(SellingPrice[i] - ProductionCost[i]) × x[i] - ActivationCost[i] × y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, production cannot exceed its maximum demand:
        -   x[i] ≤ MaximumDemand[i]
    -   Constraint 2 (Individual Production Quota): For each product i, production cannot exceed the maximum possible over 22 days at its daily quota:
        -   x[i] ≤ 22 × DailyProductionQuota[i]
    -   Constraint 3 (Minimum Batch Size): If a product is produced (y[i]=1), at least its minimum batch size must be produced:
        -   x[i] ≥ MinimumBatchSize[i] × y[i]
    -   Constraint 4 (Activation Linking): If a product is not produced (y[i]=0), its production quantity must be zero; if produced, cannot exceed demand:
        -   x[i] ≤ MaximumDemand[i] × y[i]
    -   Constraint 5 (Shared Production Days): The sum of equivalent production days used by all products cannot exceed 22:
        -   sum over i of [x[i] / DailyProductionQuota[i]] ≤ 22
    -   Constraint 6 (Variable Types): 
        -   x[i] ≥ 0 and integer for all i
        -   y[i] ∈ {0,1} for all i
[Abstract Model Plan END]