[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of their "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary index is the set of all courses in the CSV file where `discipline` = "Operations Research". Let this set be denoted as $C$ (the set of eligible Operations Research courses).
4.  **Define Decision Variables:**
    -   $x_c$ = 1 if course $c \in C$ is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `interest_points` column (the interest score for each course $c$).
    -   Constraint coefficients: `credits` column (the number of credits for each course $c$).
    -   Constraint RHS: The total credit limit is 20 (fixed constant from the query).
6.  **Formulate Objective:** Maximize the total interest points of selected courses: $\sum_{c \in C} \text{interest_points}_c \cdot x_c$.
7.  **Formulate Constraints:**
    -   Credit Limit: $\sum_{c \in C} \text{credits}_c \cdot x_c \leq 20$.
    -   Binary Selection: $x_c \in \{0,1\}$ for all $c \in C$.
[Abstract Model Plan END]