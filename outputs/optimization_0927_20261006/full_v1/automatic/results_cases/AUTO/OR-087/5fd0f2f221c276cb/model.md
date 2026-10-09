[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit while considering product-specific demand, prices, production costs, daily production quotas, fixed activation costs, and minimum batch size restrictions. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as \( i \in \{A1, A2, ..., A80\} \).
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product \( i \) to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product \( i \) is activated (i.e., any of product \( i \) is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
    -   Maximum demand per product: from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
    -   Daily production quota per product: from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
    -   Fixed activation cost per product: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Minimum batch size per product: from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
    -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of (selling price – production cost) × quantity produced, minus the fixed activation cost for each product if its line is activated:
    \[
    \text{Maximize} \quad \sum_{i} \left[ (\text{Price}_i - \text{Cost}_i) \cdot x[i] - \text{ActivationCost}_i \cdot y[i] \right]
    \]
7.  **Formulate Constraints:**
    -   **Demand Constraint:** For each product \( i \), production cannot exceed its maximum demand:
        \[
        x[i] \leq \text{MaxDemand}_i
        \]
    -   **Production Quota Constraint:** For each product \( i \), production cannot exceed the maximum possible output over 22 days:
        \[
        x[i] \leq 22 \times \text{DailyQuota}_i
        \]
    -   **Minimum Batch Size Constraint:** For each product \( i \), if the line is activated, production must be at least the minimum batch size:
        \[
        x[i] \geq \text{MinBatch}_i \cdot y[i]
        \]
    -   **Activation Linking Constraint:** For each product \( i \), production is only allowed if the line is activated:
        \[
        x[i] \leq \text{MaxDemand}_i \cdot y[i]
        \]
    -   **Shared Production Days Constraint:** The total equivalent production days used across all products cannot exceed 22:
        \[
        \sum_{i} \frac{x[i]}{\text{DailyQuota}_i} \leq 22
        \]
    -   **Variable Domains:** For all \( i \), \( x[i] \geq 0 \), integer; \( y[i] \in \{0,1\} \).
[Abstract Model Plan END]