[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine an optimal monthly production plan for 80 products over 22 days, maximizing total profit. The plan must account for product-specific maximum demand, selling prices, production costs, daily production quotas, fixed activation costs per product, and minimum batch sizes. All production and activation decisions must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products (i ∈ {A1, A2, ..., A80}).
4.  **Define Decision Variables:**
    -   `x[i]` = Total quantity of product i to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if product i is produced (i.e., its production line is activated), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Maximum demand per product: from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
    -   Selling price per product: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
    -   Production cost per product: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
    -   Daily production quota per product: from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
    -   Fixed activation cost per product: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Minimum batch size per product: from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
    -   Number of production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × produced quantity] minus the sum of fixed activation costs for each product produced:
    -   Maximize:  
        sum over i [ (SellingPrice[i] - ProductionCost[i]) × x[i] - ActivationCost[i] × y[i] ]
7.  **Formulate Constraints:**
    -   Demand constraint: For each product i, x[i] ≤ MaximumDemand[i].
    -   Production quota constraint: For each product i, x[i] ≤ DailyQuota[i] × 22.
    -   Minimum batch size constraint: For each product i, x[i] = 0 or x[i] ≥ MinBatchSize[i]. (Enforced as: x[i] ≥ MinBatchSize[i] × y[i])
    -   Linking constraint: For each product i, x[i] ≤ (MaximumDemand[i] or DailyQuota[i] × 22) × y[i] (ensures y[i] = 1 iff x[i] > 0).
    -   Integrality constraints: x[i] ∈ {0, 1, 2, ...} (integer), y[i] ∈ {0, 1} (binary).
[Abstract Model Plan END]