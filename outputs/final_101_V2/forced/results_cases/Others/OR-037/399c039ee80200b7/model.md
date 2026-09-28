[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily ordering quantities for each vehicle type (car model) to maximize total profit, while ensuring that the total inventory ordered does not exceed the company's overall stock capacity.
2.  **Identify Model Type:** Based on the query, this is a classic Integer Linear Programming (ILP) problem, specifically a multi-item knapsack problem.
3.  **Define Index Sets:** The primary index is the set of vehicle types (products), as listed in `products.csv` (e.g., Sedan, SUV, Truck, etc.).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of vehicles of type `i` to order per day. Type: GRB.INTEGER (since vehicles are indivisible).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from `products.csv` (profit per vehicle of type i).
    -   Constraint coefficients: 'Weight' column from `products.csv` (inventory space each vehicle of type i occupies).
    -   Constraint RHS: 'Capacity' value from `capacity.csv` (total inventory space available, e.g., 765 units).
6.  **Formulate Objective:** Maximize the total profit from all ordered vehicles, i.e., maximize sum over all products of (`Value[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The total inventory space used by all ordered vehicles must not exceed the overall capacity, i.e., sum over all products of (`Weight[i]` * `x[i]`) ≤ `Capacity`.
    -   Constraint 2 (Non-negativity and Integrality): For all vehicle types i, `x[i]` ≥ 0 and integer.
[Abstract Model Plan END]