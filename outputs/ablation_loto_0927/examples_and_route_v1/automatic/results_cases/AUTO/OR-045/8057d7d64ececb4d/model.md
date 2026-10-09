[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily order quantity for each type of produce, maximizing total benefit, while ensuring the total weight of all ordered produce does not exceed the supermarket's overall inventory capacity. The decision variables (order quantities) must be integers.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a classic integer knapsack/resource allocation problem.
3.  **Define Index Sets:** The primary index is the set of produce types (Products), as listed in products.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of produce type i to order daily. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (the benefit per unit of each produce type).
    -   Constraint coefficients: 'Weight' column from products.csv (the weight per unit of each produce type).
    -   Constraint RHS (limit): 'Capacity' value from capacity.csv (the total allowable weight for all produce ordered).
6.  **Formulate Objective:** Maximize the total benefit, i.e., maximize the sum over all produce types of (Value[i] * x[i]), where Value[i] is the benefit per unit of produce i.
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Capacity): The sum over all produce types of (Weight[i] * x[i]) must be less than or equal to the total capacity from capacity.csv.
    -   Constraint 2 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer (cannot order negative or fractional units).
[Abstract Model Plan END]