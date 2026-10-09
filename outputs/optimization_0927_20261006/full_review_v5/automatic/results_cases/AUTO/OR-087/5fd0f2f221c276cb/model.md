[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit by considering sales revenue, production costs, fixed activation costs, and minimum batch size restrictions, subject to demand, production quotas, and a shared production time limit. All production quantities must be integer multiples of 100 kg, and activation decisions are binary.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as `i ∈ {A1, A2, ..., A80}`.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product `i` to produce in 100 kg units. Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product `i` is activated (i.e., any of product `i` is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
    -   Maximum demand per month (100 kg units): from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
    -   Daily production quota (max per day, 100 kg units): from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
    -   Fixed activation cost per product: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Minimum batch size per product (100 kg units): from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
    -   Total available production days: 22 (from query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of (selling price - production cost) × production quantity, minus the fixed activation cost for each activated product:
    -   Objective: Maximize  
        `sum_i [ (SellingPrice[i] - ProductionCost[i]) * x[i] - ActivationCost[i] * y[i] ]`
7.  **Formulate Constraints:**
    -   Demand constraint: For each product `i`, `x[i] ≤ MaximumDemand[i]`.
    -   Production quota constraint: For each product `i`, `x[i] ≤ 22 × DailyQuota[i]` (cannot exceed what can be produced in 22 days at full capacity).
    -   Minimum batch size constraint: For each product `i`, if any is produced, must meet minimum batch: `x[i] ≥ MinimumBatch[i] * y[i]`.
    -   Activation linking constraint: For each product `i`, cannot produce unless activated: `x[i] ≤ MaximumDemand[i] * y[i]`.
    -   Shared production time constraint: The sum over all products of (production quantity ÷ daily quota) must not exceed 22 days:  
        `sum_i [ x[i] / DailyQuota[i] ] ≤ 22`
    -   Integrality and binary constraints: For all `i`, `x[i]` integer ≥ 0; `y[i]` binary (0 or 1).
[Abstract Model Plan END]