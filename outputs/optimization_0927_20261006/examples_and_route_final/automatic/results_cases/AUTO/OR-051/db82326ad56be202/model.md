[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various coffee product units into multiple retail cabinets, maximizing the total value of products placed, while ensuring that the total weight in each cabinet does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a multi-knapsack allocation problem.
3.  **Define Index Sets:** The primary indices are:
    - Cabinets (indexed by i, from 'CabinetID' in capacity.csv)
    - Products (indexed by j, from 'ProductName' in products.csv)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of coffee product j placed in cabinet i. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (value per unit of product j).
    -   Constraint coefficients: 'Weight' column from products.csv (weight per unit of product j).
    -   Constraint RHS (limits): 'Capacity' column from capacity.csv (maximum total weight allowed in cabinet i).
6.  **Formulate Objective:** Maximize the total value of all products placed across all cabinets, i.e., maximize the sum over all cabinets and products of (Value of product j) × (number of units of product j in cabinet i):  
        Maximize ∑₍i₎ ∑₍j₎ [Value[j] * x[i,j]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Cabinet Capacity): For each cabinet i, the total weight of all products placed in cabinet i must not exceed its capacity:  
        ∑₍j₎ [Weight[j] * x[i,j]] ≤ Capacity[i]
    -   Constraint 2 (Non-negativity and Integrality): For all cabinets i and products j, x[i,j] ≥ 0 and integer.
[Abstract Model Plan END]