[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the total "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, specifically a Binary Integer Program).
3.  **Define Index Sets:** The primary index is the set of all courses in the "Operations Research" discipline (let’s call this set OR_COURSES).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course i is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'interest_points' column (the interest score for each course).
    -   Constraint coefficients: 'credits' column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (given in the query).
6.  **Formulate Objective:** Maximize the sum of 'interest_points' for selected Operations Research courses, i.e., maximize sum over i in OR_COURSES of (schema['interest_points'][i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Credit Limit): The sum of 'credits' for selected Operations Research courses must not exceed 20, i.e., sum over i in OR_COURSES of (schema['credits'][i] * x[i]) ≤ 20.
    -   Constraint 2 (Selection): Each x[i] is binary (0 or 1), i.e., x[i] ∈ {0,1} for all i in OR_COURSES.
[Abstract Model Plan END]