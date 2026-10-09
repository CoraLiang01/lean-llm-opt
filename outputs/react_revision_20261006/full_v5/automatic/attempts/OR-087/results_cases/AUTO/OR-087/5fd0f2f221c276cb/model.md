[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as {A1, A2, ..., A80}.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)'.
        -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)'.
        -   Fixed activation cost: from 36-2.csv, row 'Activation Cost ($)'.
    -   Constraint coefficients:
        -   Maximum demand per product: from 36-1.csv, row 'Maximum Demand (100 kg units)'.
        -   Daily production quota per product: from 36-1.csv, row 'Production Quota (max per day)'.
        -   Minimum batch size per product: from 36-3.csv, row 'Minimum Batch Size (100 kg units)'.
    -   Shared resource:
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for each product that is produced:
    -   Maximize:  
        sum over i of [(Selling Price[i] - Production Cost[i]) × x[i] - Activation Cost[i] × y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i, x[i] ≤ Maximum Demand[i].
    -   Constraint 2 (Production Quota Limit): For each product i, x[i] ≤ 22 × Production Quota[i] (cannot exceed what could be produced if all 22 days were dedicated to product i).
    -   Constraint 3 (Minimum Batch Size): For each product i, if product i is produced, x[i] ≥ Minimum Batch Size[i] × y[i].
    -   Constraint 4 (Activation Linking): For each product i, x[i] ≤ Maximum Demand[i] × y[i] (or a similar large upper bound, ensuring x[i] = 0 if y[i] = 0).
    -   Constraint 5 (Shared Production Days): The sum over all products of (x[i] / Production Quota[i]) ≤ 22 (i.e., total equivalent production days used across all products cannot exceed 22).
    -   Constraint 6 (Integrality): For all i, x[i] ≥ 0 and integer; y[i] ∈ {0,1}.
[Abstract Model Plan END]