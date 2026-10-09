[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal production plan for 80 products over a 22-day month, maximizing total profit. The plan must account for product-specific maximum demand, selling price, production cost, daily production quota, fixed activation cost, and minimum batch size. Production quantities are integer multiples of 100 kg, and each product's production line can be activated or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation cost) and batch-size constraints.
3.  **Define Index Sets:** The primary index is the set of Products, denoted as $i \in \{\text{A1}, \text{A2}, ..., \text{A80}\}$.
4.  **Define Decision Variables:**
    -   $x[i]$ = Quantity of product $i$ to produce (in units of 100 kg). Type: GRB.INTEGER.
    -   $y[i]$ = 1 if the production line for product $i$ is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Selling price per 100 kg: from 36-1.csv, row "Selling Price ($/100 kg)", columns A1–A80.
    -   Production cost per 100 kg: from 36-1.csv, row "Production Cost ($/100 kg)", columns A1–A80.
    -   Maximum demand (100 kg units): from 36-1.csv, row "Maximum Demand (100 kg units)", columns A1–A80.
    -   Daily production quota (max per day): from 36-1.csv, row "Production Quota (max per day)", columns A1–A80.
    -   Fixed activation cost: from 36-2.csv, row "Activation Cost ($)", columns A1–A80.
    -   Minimum batch size (100 kg units): from 36-3.csv, row "Minimum Batch Size (100 kg units)", columns A1–A80.
    -   Total available production days: 22 (given in query).
6.  **Formulate Objective:** Maximize total profit, defined as the sum over all products of [(selling price – production cost) × production quantity] minus the sum of fixed activation costs for activated lines:
    $$
    \text{Maximize} \quad \sum_{i} \left( (\text{Selling Price}_i - \text{Production Cost}_i) \cdot x[i] - \text{Activation Cost}_i \cdot y[i] \right)
    $$
7.  **Formulate Constraints:**
    -   **Demand Constraint:** For each product $i$, $x[i] \leq \text{Maximum Demand}_i$.
    -   **Production Quota Constraint:** For each product $i$, $x[i] \leq 22 \times \text{Production Quota}_i$ (cannot exceed what can be produced in 22 days if fully dedicated).
    -   **Minimum Batch Size Constraint:** For each product $i$, $x[i] \geq \text{Minimum Batch Size}_i \cdot y[i]$ (if activated, must produce at least the minimum batch).
    -   **Activation Linking Constraint:** For each product $i$, $x[i] \leq \text{Maximum Demand}_i \cdot y[i]$ (cannot produce unless activated).
    -   **Shared Production Days Constraint:** The sum over all products of (production quantity / daily quota) must not exceed 22:
        $$
        \sum_{i} \frac{x[i]}{\text{Production Quota}_i} \leq 22
        $$
    -   **Variable Domains:** $x[i] \geq 0$, integer; $y[i] \in \{0,1\}$ for all $i$.
[Abstract Model Plan END]