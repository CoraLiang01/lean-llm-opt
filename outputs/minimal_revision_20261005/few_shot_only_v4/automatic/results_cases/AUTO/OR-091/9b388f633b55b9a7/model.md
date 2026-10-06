[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of "interest points" for the selected courses is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack problem (a special case of Mixed Integer Programming, specifically a Binary Integer Program).
3.  **Define Index Sets:** The primary index is the set of Operations Research courses, i.e., all rows in the CSV where `discipline` = "Operations Research".
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'interest_points' column for each Operations Research course.
    -   Constraint coefficients: 'credits' column for each Operations Research course.
    -   Constraint RHS: The total credit limit is 20 (a constant from the query).
6.  **Formulate Objective:** Maximize the total interest points of selected Operations Research courses, i.e., maximize sum over all selected courses of `interest_points[i] * x[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Credit Limit): The sum of credits for selected Operations Research courses must not exceed 20, i.e., sum over all selected courses of `credits[i] * x[i] <= 20`.
    -   Constraint 2 (Binary Selection): For each Operations Research course `i`, `x[i]` ∈ {0, 1}.
[Abstract Model Plan END]