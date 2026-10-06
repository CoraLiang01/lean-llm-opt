[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantities for each type of bread to maximize total expected profit, subject to a storage capacity constraint. Each bread type has an associated expected profit and storage weight, and the total storage used by all ordered bread must not exceed the bakery's daily storage capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a classic 0-1 or bounded integer knapsack problem.
3.  **Define Index Sets:** The primary index is the set of bread types (Products), as listed in products.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of bread type i to order each day. Type: GRB.INTEGER (must be integer-valued).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (expected profit per unit of bread type i).
    -   Constraint coefficients: 'Weight' column from products.csv (storage space required per unit of bread type i).
    -   Constraint RHS: 'Capacity' value from capacity.csv (total available storage space per day).
6.  **Formulate Objective:** Maximize the total expected profit, i.e., maximize sum over all bread types i of (Value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Storage Capacity): The total storage used by all ordered bread must not exceed the available capacity, i.e., sum over all i of (Weight[i] * x[i]) ≤ Capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer (cannot order negative or fractional units of bread).
[Abstract Model Plan END]