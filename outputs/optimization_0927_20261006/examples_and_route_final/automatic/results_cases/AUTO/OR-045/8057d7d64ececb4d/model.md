[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantities for each type of produce to maximize total benefit, subject to an overall inventory weight capacity constraint. Each produce type has an associated weight and benefit per unit, and the total weight of all ordered units must not exceed the given capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a bounded integer knapsack problem).
3.  **Define Index Sets:** The primary index is the set of produce types, denoted as Products (from 'ProductName' in products.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of produce type i to order daily. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (benefit per unit) will come from column: 'Value' in products.csv.
    -   Constraint coefficients (weight per unit) will come from: 'Weight' in products.csv.
    -   Constraint RHS (total capacity limit) will come from: 'Capacity' in capacity.csv.
6.  **Formulate Objective:** Maximize the total benefit, i.e., maximize sum over all products i of (Value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The total weight of all ordered units must not exceed the overall capacity, i.e., sum over all products i of (Weight[i] * x[i]) ≤ Capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all products i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]