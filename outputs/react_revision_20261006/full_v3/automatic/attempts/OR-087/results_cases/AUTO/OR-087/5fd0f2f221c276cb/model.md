[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as \( i \in \{A1, A2, ..., A80\} \).
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product \( i \) to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product \( i \) is activated (i.e., product \( i \) is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)'.
        -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)'.
        -   Fixed activation cost: from 36-2.csv, row 'Activation Cost ($)'.
    -   Constraint coefficients:
        -   Maximum demand per product: from 36-1.csv, row 'Maximum Demand (100 kg units)'.
        -   Daily production quota per product: from 36-1.csv, row 'Production Quota (max per day)'.
        -   Minimum batch size per product: from 36-3.csv, row 'Minimum Batch Size (100 kg units)'.
    -   Constraint RHS:
        -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for each product that is produced:
    -   Objective: Maximize \( \sum_{i} \left[ (\text{Price}_i - \text{Cost}_i) \cdot x[i] - \text{ActivationCost}_i \cdot y[i] \right] \)
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product \( i \), \( x[i] \leq \text{MaxDemand}_i \).
    -   Constraint 2 (Production Quota Limit): For each product \( i \), \( x[i] \leq 22 \times \text{DailyQuota}_i \) (cannot produce more than the maximum possible in 22 days).
    -   Constraint 3 (Minimum Batch Size): For each product \( i \), if produced, must meet minimum batch: \( x[i] \geq \text{MinBatch}_i \cdot y[i] \).
    -   Constraint 4 (Activation Linking): For each product \( i \), \( x[i] \leq \text{MaxDemand}_i \cdot y[i] \) (cannot produce unless activated).
    -   Constraint 5 (Shared Production Days): The sum of equivalent production days used across all products cannot exceed 22:
        -   \( \sum_{i} \frac{x[i]}{\text{DailyQuota}_i} \leq 22 \)
    -   Constraint 6 (Integrality): \( x[i] \) are integer, \( y[i] \) are binary for all \( i \).
[Abstract Model Plan END]