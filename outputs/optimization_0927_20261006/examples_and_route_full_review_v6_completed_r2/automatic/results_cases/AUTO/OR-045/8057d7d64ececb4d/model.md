[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantity for each type of produce to maximize total benefit, subject to an overall inventory weight capacity constraint. Each produce type has an associated benefit (value) and weight per unit, and the total weight of all ordered units must not exceed the specified capacity. The order quantities must be integers.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a bounded integer knapsack problem.
3.  **Define Index Sets:** The primary index is the set of produce types, denoted as Products (from the 'ProductName' column in products.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of produce type i to order daily. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (benefit per unit) will come from the 'Value' column in products.csv.
    -   Constraint coefficients (weight per unit) will come from the 'Weight' column in products.csv.
    -   Constraint RHS (total capacity limit) will come from the 'Capacity' column in capacity.csv.
6.  **Formulate Objective:** Maximize the total benefit, i.e., maximize the sum over all products of (Value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The sum over all products of (Weight[i] * x[i]) must be less than or equal to the total capacity from capacity.csv.
    -   Constraint 2 (Non-negativity and Integrality): For each product i, x[i] must be an integer and x[i] ≥ 0.
[Abstract Model Plan END]