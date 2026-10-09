[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantities for each type of bread to maximize total expected profit, subject to a storage capacity constraint. Each bread type has an associated expected profit and storage weight per unit.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a 0-1 or bounded integer knapsack problem).
3.  **Define Index Sets:** The primary index is the set of bread types (Products), as listed in products.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of bread type `i` to order each day. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (expected profit per unit) will come from column: 'Value' in products.csv.
    -   Constraint coefficients (storage weight per unit) will come from column: 'Weight' in products.csv.
    -   Constraint RHS (total storage limit) will come from: 'Capacity' in capacity.csv.
6.  **Formulate Objective:** Maximize the total expected profit, i.e., maximize the sum over all bread types of (Value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Storage Capacity): The sum over all bread types of (Weight[i] * x[i]) must be less than or equal to the total storage capacity (Capacity).
    -   Constraint 2 (Non-negativity and Integrality): For each bread type, x[i] ≥ 0 and integer.
[Abstract Model Plan END]