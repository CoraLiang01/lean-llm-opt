[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of their "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack problem (a special case of Mixed Integer Programming with binary variables).
3.  **Define Index Sets:** The primary index is the set of all courses in the CSV file where `discipline` equals "Operations Research". Let this set be denoted as $C$ (the set of eligible Operations Research courses).
4.  **Define Decision Variables:**
    -   `x[c]` = 1 if course $c \in C$ is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `interest_points` column (the interest score for each course).
    -   Constraint coefficients: `credits` column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (fixed constant from the query).
6.  **Formulate Objective:** Maximize the sum of `interest_points[c] * x[c]` over all $c \in C$ (i.e., maximize total interest points from selected Operations Research courses).
7.  **Formulate Constraints:**
    -   Credit Limit: The sum of `credits[c] * x[c]` over all $c \in C$ must be less than or equal to 20.
    -   Binary Selection: For each $c \in C$, $x[c] \in \{0,1\}$ (each course is either selected or not).
[Abstract Model Plan END]