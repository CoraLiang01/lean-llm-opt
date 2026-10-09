[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantities for each type of bread to maximize total expected profit, subject to a storage capacity constraint. The decision variables are the integer number of units to order for each bread type.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a classic integer knapsack problem.
3.  **Define Index Sets:** The primary index is the set of bread types (Products), as listed in products.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of bread type i to order each day. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (profit per unit) will come from column: 'Value' in products.csv.
    -   Constraint coefficients (storage space per unit) will come from: 'Weight' in products.csv.
    -   Constraint RHS (total storage limit) will come from: 'Capacity' in capacity.csv.
6.  **Formulate Objective:** Maximize the total expected profit, i.e., maximize sum over all bread types i of (Value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Storage Capacity): The total storage used by all ordered bread must not exceed the available capacity, i.e., sum over all i of (Weight[i] * x[i]) ≤ Capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]