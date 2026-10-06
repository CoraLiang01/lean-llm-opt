[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to serve all customers, minimizing the total cost, which includes both the fixed annual opening costs of warehouses and the variable transportation costs for fulfilling all customer demand. Each warehouse has a maximum service capacity, and all customer demands must be satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (indexed by i), from 'Warehouse ID' in warehouse.csv and cost.csv.
    - Customers (indexed by j), from 'Customer ID' in demand.csv and columns C1–C20 in cost.csv.
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if warehouse i is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = amount of customer j’s demand served from warehouse i. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: 'Fixed_Cost' column in warehouse.csv.
    -   Warehouse capacity: 'Capacity' column in warehouse.csv.
    -   Customer demand: 'Demand' column in demand.csv.
    -   Transportation cost per unit from warehouse i to customer j: cost.csv, columns C1–C20 for each warehouse row.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - Fixed opening costs for all selected warehouses: sum over i of (Fixed_Cost[i] * y[i])
    - Plus total transportation costs: sum over i and j of (cost[i][j] * x[i,j])
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer j, the sum over all warehouses i of x[i,j] must equal the total demand of customer j (i.e., all demand must be met, and each customer can be served by one or more warehouses).
    -   Constraint 2 (Warehouse Capacity): For each warehouse i, the sum over all customers j of x[i,j] must not exceed the capacity of warehouse i.
    -   Constraint 3 (Linking): For each warehouse i and customer j, x[i,j] ≤ Demand[j] * y[i] (or, more generally, x[i,j] ≤ Capacity[i] * y[i]), ensuring that no demand is assigned to a warehouse unless it is open.
    -   Constraint 4 (Non-negativity): All x[i,j] ≥ 0.
    -   Constraint 5 (Binary): All y[i] ∈ {0,1}.
[Abstract Model Plan END]