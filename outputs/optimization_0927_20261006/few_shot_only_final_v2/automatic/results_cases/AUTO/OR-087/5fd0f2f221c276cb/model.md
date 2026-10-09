[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal monthly production plan for 80 products, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quota, fixed activation costs, and minimum batch sizes. Production is limited to 22 days, and all variables must be integer (quantities in 100 kg units, activation as binary).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as \( i \in \{A1, A2, ..., A80\} \).
4.  **Define Decision Variables:**
    -   `x[i]` = Quantity of product \( i \) to produce in the month (in integer multiples of 100 kg). Type: GRB.INTEGER.
    -   `y[i]` = 1 if production line for product \( i \) is activated (i.e., any of product \( i \) is produced), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Selling Price: from "Selling Price ($/100 kg)" in 36-1.csv.
        -   Production Cost: from "Production Cost ($/100 kg)" in 36-1.csv.
        -   Fixed Activation Cost: from "Activation Cost ($)" in 36-2.csv.
    -   Constraint coefficients:
        -   Maximum Demand: from "Maximum Demand (100 kg units)" in 36-1.csv.
        -   Daily Production Quota: from "Production Quota (max per day)" in 36-1.csv.
        -   Minimum Batch Size: from "Minimum Batch Size (100 kg units)" in 36-3.csv.
        -   Total Production Days: fixed at 22 (shared across all products).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(Selling Price - Production Cost) × quantity produced] minus the sum of fixed activation costs for each activated product line:
    \[
    \text{Maximize} \quad \sum_{i} \left( (\text{Selling Price}_i - \text{Production Cost}_i) \cdot x[i] - \text{Activation Cost}_i \cdot y[i] \right)
    \]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each product \( i \), \( x[i] \leq \) Maximum Demand\(_i\).
    -   Constraint 2 (Production Quota): For each product \( i \), \( x[i] \leq \) (Daily Production Quota\(_i\) × 22).
    -   Constraint 3 (Minimum Batch Size): For each product \( i \), \( x[i] \geq \) Minimum Batch Size\(_i\) × \( y[i] \).
    -   Constraint 4 (Activation Linking): For each product \( i \), \( x[i] \leq \) Maximum Demand\(_i\) × \( y[i] \) (ensures \( x[i]=0 \) if not activated).
    -   Constraint 5 (Total Production Days): The sum over all products of (quantity produced ÷ daily production quota) must not exceed 22:
        \[
        \sum_{i} \frac{x[i]}{\text{Daily Production Quota}_i} \leq 22
        \]
    -   Constraint 6 (Integrality): For all \( i \), \( x[i] \geq 0 \), integer; \( y[i] \in \{0,1\} \).
[Abstract Model Plan END]