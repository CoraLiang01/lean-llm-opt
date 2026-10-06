[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of "interest points" for the selected courses is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, specifically a Binary Integer Program).
3.  **Define Index Sets:** The primary index is the set of all courses in the "Operations Research" discipline. Let \( i \) index these courses.
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course \( i \) is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'interest_points' column (the interest score for each course).
    -   Constraint coefficients: 'credits' column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (given in the query).
    -   Data selection: Only rows where 'discipline' == "Operations Research" are included in the model.
6.  **Formulate Objective:** Maximize the total interest points of selected Operations Research courses, i.e., maximize sum over \( i \) of `interest_points[i] * x[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Credit Limit): The sum over \( i \) of `credits[i] * x[i]` ≤ 20.
    -   Constraint 2 (Binary Selection): For each course \( i \), `x[i]` ∈ {0, 1}.
[Abstract Model Plan END]