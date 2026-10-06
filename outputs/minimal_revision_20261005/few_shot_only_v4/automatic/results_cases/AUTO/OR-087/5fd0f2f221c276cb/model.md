[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch size constraints.
3.  **Define Index Sets:** The primary index is Products, specifically the set {A1, A2, ..., A80}.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product i to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product i is activated (i.e., product i is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Revenue per unit: 'Selling Price ($/100 kg)' from 36-1.csv.
        -   Variable production cost per unit: 'Production Cost ($/100 kg)' from 36-1.csv.
        -   Fixed activation cost: 'Activation Cost ($)' from 36-2.csv.
    -   Constraint coefficients:
        -   Maximum demand: 'Maximum Demand (100 kg units)' from 36-1.csv.
        -   Daily production quota: 'Production Quota (max per day)' from 36-1.csv.
        -   Minimum batch size: 'Minimum Batch Size (100 kg units)' from 36-3.csv.
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of [(Selling Price - Production Cost) × quantity produced] minus the sum of fixed activation costs for each activated product line:
    -   Maximize:  
        sum over i of [(Selling Price[i] - Production Cost[i]) × x[i] - Activation Cost[i] × y[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product i,  
        x[i] ≤ Maximum Demand[i]
    -   Constraint 2 (Production Quota Limit): For each product i,  
        x[i] ≤ 22 × Production Quota[i]  
        (since at most 22 days of production are available)
    -   Constraint 3 (Minimum Batch Size): For each product i,  
        x[i] ≥ Minimum Batch Size[i] × y[i]  
        (if a product is produced, at least the minimum batch size must be met; if not produced, x[i]=0)
    -   Constraint 4 (Activation Linking): For each product i,  
        x[i] ≤ Maximum Demand[i] × y[i]  
        (if y[i]=0, x[i]=0; if y[i]=1, x[i] can be up to demand)
    -   Constraint 5 (Shared Production Days):  
        sum over i of [x[i] / Production Quota[i]] ≤ 22  
        (total equivalent production days used across all products cannot exceed 22)
    -   Constraint 6 (Integrality):  
        x[i] ≥ 0 and integer for all i  
        y[i] ∈ {0,1} for all i
[Abstract Model Plan END]