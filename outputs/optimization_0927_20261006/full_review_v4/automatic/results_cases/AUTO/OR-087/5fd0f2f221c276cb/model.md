[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quota, fixed activation costs, and minimum batch sizes. Production is limited to 22 days per month, and all decision variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as $i \in \{\text{A1}, \text{A2}, ..., \text{A80}\}$.
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product $i$ to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product $i$ is activated (i.e., any of product $i$ is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row "Selling Price ($/100 kg)", columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row "Production Cost ($/100 kg)", columns A1–A80.
    -   Maximum demand (100 kg units): from 36-1.csv, row "Maximum Demand (100 kg units)", columns A1–A80.
    -   Daily production quota (max per day): from 36-1.csv, row "Production Quota (max per day)", columns A1–A80.
    -   Fixed activation cost: from 36-2.csv, row "Activation Cost ($)", columns A1–A80.
    -   Minimum batch size (100 kg units): from 36-3.csv, row "Minimum Batch Size (100 kg units)", columns A1–A80.
    -   Number of production days: fixed at 22 (from query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price – production cost) × quantity produced] minus the sum of fixed activation costs for each activated product line:
    -   $\max \sum_{i} \left[(\text{Selling Price}_i - \text{Production Cost}_i) \cdot x[i] - \text{Activation Cost}_i \cdot y[i]\right]$
7.  **Formulate Constraints:**
    -   **Demand Constraint:** For each product $i$, $x[i] \leq$ Maximum Demand$_i$.
    -   **Production Quota Constraint:** For each product $i$, $x[i] \leq$ (Daily Production Quota$_i$ × 22).
    -   **Minimum Batch Size Constraint:** For each product $i$, $x[i] \geq$ (Minimum Batch Size$_i$ × $y[i]$).
    -   **Activation Linking Constraint:** For each product $i$, $x[i] \leq$ (Maximum Demand$_i$ × $y[i]$) (ensures $x[i]=0$ if $y[i]=0$).
    -   **Aggregate Production Days Constraint:** The sum over all products of (quantity produced ÷ daily production quota) must not exceed 22 days:
        -   $\sum_{i} \frac{x[i]}{\text{Daily Production Quota}_i} \leq 22$
    -   **Variable Domains:** $x[i] \geq 0$, integer; $y[i] \in \{0,1\}$ for all $i$.
[Abstract Model Plan END]