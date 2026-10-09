[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quotas, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as \( i \in \{A1, A2, ..., A80\} \).
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product \( i \) to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product \( i \) is activated (i.e., any of product \( i \) is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row 'Selling Price ($/100 kg)', columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row 'Production Cost ($/100 kg)', columns A1–A80.
    -   Maximum demand (100 kg units): from 36-1.csv, row 'Maximum Demand (100 kg units)', columns A1–A80.
    -   Daily production quota (max per day): from 36-1.csv, row 'Production Quota (max per day)', columns A1–A80.
    -   Fixed activation cost: from 36-2.csv, row 'Activation Cost ($)', columns A1–A80.
    -   Minimum batch size (100 kg units): from 36-3.csv, row 'Minimum Batch Size (100 kg units)', columns A1–A80.
    -   Number of production days available: 22 (from query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price - production cost) × quantity produced] minus the sum of fixed activation costs for each activated product line:
    -   Objective: Maximize  
        \[
        \sum_{i} \left[ (\text{Selling Price}_i - \text{Production Cost}_i) \times x[i] - \text{Activation Cost}_i \times y[i] \right]
        \]
7.  **Formulate Constraints:**
    -   **Demand Constraint:** For each product \( i \), production cannot exceed maximum demand:
        -   \( x[i] \leq \text{Maximum Demand}_i \)
    -   **Production Quota Constraint:** For each product \( i \), production cannot exceed what is possible in 22 days at the product's daily quota:
        -   \( x[i] \leq 22 \times \text{Production Quota}_i \)
    -   **Minimum Batch Size Constraint:** If a product is produced (i.e., \( y[i] = 1 \)), at least the minimum batch size must be produced:
        -   \( x[i] \geq \text{Minimum Batch Size}_i \times y[i] \)
    -   **Activation Linking Constraint:** If a product is not activated (\( y[i] = 0 \)), no production is allowed:
        -   \( x[i] \leq \text{Maximum Demand}_i \times y[i] \) (or any sufficiently large upper bound, but using demand is tightest)
    -   **Shared Production Days Constraint:** The total equivalent production days used across all products cannot exceed 22. For each product, producing one unit (100 kg) uses \( 1/\text{Production Quota}_i \) days:
        -   \( \sum_{i} \frac{x[i]}{\text{Production Quota}_i} \leq 22 \)
    -   **Variable Domains:** For all \( i \):
        -   \( x[i] \geq 0 \), integer
        -   \( y[i] \in \{0, 1\} \)
[Abstract Model Plan END]