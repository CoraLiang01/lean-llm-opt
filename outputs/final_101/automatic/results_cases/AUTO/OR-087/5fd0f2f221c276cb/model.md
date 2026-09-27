[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as `i ∈ {A1, A2, ..., A80}`.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product `i` to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product `i` is activated (i.e., any of product `i` is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
        -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
        -   Fixed activation cost: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Constraint coefficients:
        -   Maximum demand: from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
        -   Daily production quota: from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
        -   Minimum batch size: from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for each product that is produced. That is:  
    Maximize  
    `sum_i [ (SellingPrice[i] - ProductionCost[i]) * x[i] - ActivationCost[i] * y[i] ]`
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product `i`, `x[i] ≤ MaximumDemand[i]` (do not exceed market demand).
    -   Constraint 2 (Individual Production Quota): For each product `i`, `x[i] ≤ 22 × DailyProductionQuota[i]` (cannot produce more than the line's capacity over 22 days).
    -   Constraint 3 (Minimum Batch Size): For each product `i`, if any is produced, it must be at least the minimum batch size: `x[i] ≥ MinimumBatchSize[i] * y[i]`.
    -   Constraint 4 (Activation Linking): For each product `i`, only allow production if the line is activated: `x[i] ≤ MaximumDemand[i] * y[i]` (or tighter: min of demand and 22×quota).
    -   Constraint 5 (Shared Production Days): The sum over all products of (quantity produced ÷ daily quota) must not exceed 22 days:  
        `sum_i [ x[i] / DailyProductionQuota[i] ] ≤ 22`
    -   Constraint 6 (Integrality): All `x[i]` are integer ≥ 0; all `y[i]` are binary (0 or 1).
[Abstract Model Plan END]