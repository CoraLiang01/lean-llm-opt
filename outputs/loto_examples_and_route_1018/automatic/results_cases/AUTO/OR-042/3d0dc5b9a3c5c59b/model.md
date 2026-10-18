[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantity for each drug type to maximize total benefit, subject to a single overall inventory weight capacity. Each drug type has a benefit coefficient and a per-unit weight; the total weight of all ordered drugs must not exceed the given capacity. The order quantities must be integer values.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 or general integer knapsack problem.
3.  **Define Index Sets:** The primary index is the set of drug types, denoted as `i ∈ Products`, where each product corresponds to a row in `products.csv`.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of drug type `i` to order daily. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (benefit per unit) come from column: `'Value'` in `products.csv`.
    -   Constraint coefficients (weight per unit) come from column: `'Weight'` in `products.csv`.
    -   Constraint RHS (total capacity) comes from column: `'Capacity'` in `capacity.csv`.
6.  **Formulate Objective:** Maximize the total benefit, i.e., maximize the sum over all drug types of (`Value[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The sum over all drug types of (`Weight[i]` * `x[i]`) must be less than or equal to the total capacity from `capacity.csv`.
    -   Constraint 2 (Non-negativity and Integrality): For all drug types `i`, `x[i]` must be integer and greater than or equal to zero.
[Abstract Model Plan END]