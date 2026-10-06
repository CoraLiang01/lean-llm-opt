[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables are integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as `i ∈ {A1, A2, ..., A80}`.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product `i` to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product `i` is activated (i.e., product `i` is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
    -   Maximum demand (100 kg units): from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
    -   Daily production quota (max per day): from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
    -   Fixed activation cost: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Minimum batch size (100 kg units): from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
    -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as:
    -   Total revenue from all products: sum over i of (Selling Price[i] * x[i])
    -   Minus total variable production costs: sum over i of (Production Cost[i] * x[i])
    -   Minus total fixed activation costs: sum over i of (Activation Cost[i] * y[i])
    -   Objective: Maximize sum over i of [(Selling Price[i] - Production Cost[i]) * x[i] - Activation Cost[i] * y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, `x[i] <= Maximum Demand[i]`
    -   Constraint 2 (Production Quota Limit): For each product i, `x[i] <= 22 * Daily Production Quota[i]` (cannot produce more than the line's daily quota times available days)
    -   Constraint 3 (Minimum Batch Size): For each product i, `x[i] >= Minimum Batch Size[i] * y[i]` (if produced, must meet minimum batch)
    -   Constraint 4 (Activation Linking): For each product i, `x[i] <= Maximum Demand[i] * y[i]` (if not activated, cannot produce)
    -   Constraint 5 (Total Production Days): The sum over all products of (x[i] / Daily Production Quota[i]) <= 22 (total equivalent production days used cannot exceed available days)
    -   Constraint 6 (Integrality): For all i, `x[i]` is integer ≥ 0; `y[i]` is binary (0 or 1)
[Abstract Model Plan END]