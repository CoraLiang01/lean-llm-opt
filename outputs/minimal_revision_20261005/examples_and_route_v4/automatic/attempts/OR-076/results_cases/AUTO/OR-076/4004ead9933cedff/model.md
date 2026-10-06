[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demands to these warehouses, so that the total cost (fixed warehouse opening costs plus variable transportation costs) is minimized. Each warehouse has a maximum capacity, and all customer demands must be fully satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from 'Warehouse ID' in warehouse.csv and cost.csv): W1, W2, ..., W10
    - Customers (from 'Customer ID' in demand.csv and columns C1–C20 in cost.csv): C1, C2, ..., C20
4.  **Define Decision Variables:**
    -   `y[w]` = 1 if warehouse w is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[w,c]` = amount of customer c's demand served by warehouse w. Type: GRB.CONTINUOUS (≥0).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: 'Fixed_Cost' column in warehouse.csv.
    -   Warehouse capacity: 'Capacity' column in warehouse.csv.
    -   Customer demand: 'Demand' column in demand.csv.
    -   Transportation cost per unit from warehouse w to customer c: cost.csv, entry at row 'Warehouse ID' = w, column c (e.g., 'C1', 'C2', ...).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for selected warehouses plus the sum of transportation costs for all customer assignments:
        - Objective = sum over w of (Fixed_Cost[w] * y[w]) + sum over w,c of (cost[w][c] * x[w,c])
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c, the sum of x[w,c] over all warehouses w must equal the total demand of customer c (i.e., all demand must be met, and can be split among warehouses):
            sum over w of x[w,c] = Demand[c]   for all c
    -   Constraint 2 (Warehouse Capacity): For each warehouse w, the total amount assigned from w to all customers cannot exceed its capacity, and only if the warehouse is open:
            sum over c of x[w,c] ≤ Capacity[w] * y[w]   for all w
    -   Constraint 3 (Non-negativity): x[w,c] ≥ 0 for all w, c
    -   Constraint 4 (Binary): y[w] ∈ {0,1} for all w
[Abstract Model Plan END]