[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of "interest points" for the selected courses is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, specifically a Binary Integer Program).
3.  **Define Index Sets:** The primary index is the set of all courses in the "Operations Research" discipline. Let \( i \) index these courses.
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course \( i \) is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'interest_points' column (the interest score for each course).
    -   Constraint coefficients: 'credits' column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (a fixed value from the query, not from the data).
    -   Data selection: Only rows where 'discipline' == "Operations Research" are included; all other courses are excluded.
6.  **Formulate Objective:** Maximize the total interest points of selected Operations Research courses: sum over all selected courses of their 'interest_points'.
7.  **Formulate Constraints:**
    -   Constraint 1 (Credit Limit): The sum of 'credits' for selected Operations Research courses must not exceed 20; i.e., sum over all selected courses of their 'credits' ≤ 20.
    -   Constraint 2 (Selection): Each course can be either selected or not selected; i.e., \( x[i] \in \{0,1\} \) for each Operations Research course \( i \).
[Abstract Model Plan END]