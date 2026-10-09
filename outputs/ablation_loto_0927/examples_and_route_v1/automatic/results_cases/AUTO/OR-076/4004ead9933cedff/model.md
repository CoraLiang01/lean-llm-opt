[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demands to these warehouses, such that the total cost (fixed warehouse opening costs plus variable transportation costs) is minimized. Each warehouse has a maximum capacity, and all customer demands must be fully satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from 'Warehouse ID' in warehouse.csv and cost.csv; 10 total: W1–W10)
    - Customers (from 'Customer ID' in demand.csv and cost.csv; 20 total: C1–C20)
4.  **Define Decision Variables:**
    -   `y[w]` = 1 if warehouse w is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[w,c]` = amount of customer c's demand served from warehouse w. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: 'Fixed_Cost' column in warehouse.csv.
    -   Warehouse capacity: 'Capacity' column in warehouse.csv.
    -   Customer demand: 'Demand' column in demand.csv.
    -   Transportation cost per unit from warehouse w to customer c: cost.csv, columns 'C1'–'C20' for each warehouse row.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - Fixed opening costs for all opened warehouses: sum over w of (Fixed_Cost[w] * y[w])
    - Transportation costs for all assignments: sum over w,c of (cost[w][c] * x[w,c])
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c, the sum over all warehouses w of x[w,c] must equal the total demand of customer c (i.e., all demand must be assigned).
    -   Constraint 2 (Warehouse Capacity): For each warehouse w, the sum over all customers c of x[w,c] must not exceed the capacity of warehouse w, and can only be positive if y[w] = 1 (i.e., x[w,c] ≤ Capacity[w] * y[w]).
    -   Constraint 3 (Assignment Feasibility): x[w,c] ≥ 0 for all w, c.
    -   Constraint 4 (Warehouse Opening): y[w] ∈ {0,1} for all w.
[Abstract Model Plan END]