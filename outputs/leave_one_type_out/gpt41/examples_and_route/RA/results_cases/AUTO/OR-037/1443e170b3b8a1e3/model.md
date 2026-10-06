[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily ordering quantities for each vehicle type (car model) to maximize total profit, while ensuring that the total inventory ordered does not exceed the company's overall stock capacity.
2.  **Identify Model Type:** Based on the query, this is a classic Integer Linear Programming (ILP) problem, specifically a multi-item knapsack problem.
3.  **Define Index Sets:** The primary index is the set of vehicle types (products), as listed in `products.csv`.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of vehicles of type `i` to order per day. Type: GRB.INTEGER (since vehicles are indivisible).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (profit per vehicle) will come from column: `'Value'` in `products.csv`.
    -   Constraint coefficients (space/weight per vehicle) will come from column: `'Weight'` in `products.csv`.
    -   Constraint RHS (total inventory capacity) will come from: `'Capacity'` in `capacity.csv`.
6.  **Formulate Objective:** Maximize the total profit from all ordered vehicles, i.e., maximize the sum over all vehicle types of (`Value[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The total space/weight of all ordered vehicles must not exceed the overall capacity, i.e., sum over all vehicle types of (`Weight[i]` * `x[i]`) ≤ `Capacity`.
    -   Constraint 2 (Non-negativity and Integrality): For all vehicle types, `x[i]` ≥ 0 and integer.
[Abstract Model Plan END]