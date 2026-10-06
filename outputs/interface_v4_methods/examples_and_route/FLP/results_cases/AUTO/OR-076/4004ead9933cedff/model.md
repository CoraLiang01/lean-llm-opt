[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demands to these warehouses, so that the total cost (fixed warehouse opening costs plus variable transportation costs) is minimized. Each warehouse has a maximum capacity, and all customer demands must be fully satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from 'Warehouse ID' in warehouse.csv and cost.csv): W1, W2, ..., W10
    - Customers (from 'Customer ID' in demand.csv and columns C1–C20 in cost.csv): C1, C2, ..., C20
4.  **Define Decision Variables:**
    -   `y[w]` = 1 if warehouse w is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[w,c]` = amount of customer c's demand served by warehouse w. Type: GRB.CONTINUOUS (non-negative, up to customer demand).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: 'Fixed_Cost' column in warehouse.csv, indexed by 'Warehouse ID'.
    -   Warehouse capacity: 'Capacity' column in warehouse.csv, indexed by 'Warehouse ID'.
    -   Customer demand: 'Demand' column in demand.csv, indexed by 'Customer ID'.
    -   Transportation cost per unit from warehouse w to customer c: cost.csv, entry at row 'Warehouse ID' = w, column c (e.g., 'C1', 'C2', ...).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - Fixed opening costs for all opened warehouses: sum over w of Fixed_Cost[w] * y[w]
    - Plus total transportation costs: sum over all w and c of cost[w][c] * x[w,c]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c, the sum over all warehouses w of x[w,c] must equal the total demand of customer c (i.e., all demand must be met, and can be split among warehouses).
    -   Constraint 2 (Warehouse Capacity): For each warehouse w, the sum over all customers c of x[w,c] must be less than or equal to the capacity of warehouse w (i.e., total assigned demand cannot exceed warehouse capacity).
    -   Constraint 3 (Linking): For each warehouse w and customer c, x[w,c] ≤ demand[c] * y[w] (i.e., a warehouse can only serve customers if it is open; if y[w]=0, then x[w,c]=0).
    -   Constraint 4 (Non-negativity): All x[w,c] ≥ 0.
    -   Constraint 5 (Binary): All y[w] ∈ {0,1}.
[Abstract Model Plan END]